#include <string>
#include <iostream>
#include <vector>
#include <fstream>

#include <folly/String.h>
#include <folly/Conv.h>
#include <folly/Optional.h>

#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/operator_gradient.h>

#include "embedding.hpp"

using folly::Optional;

struct Example
{
    std::vector<std::string> word, pos, label;
    std::vector<int> head;
};

auto read_conll(const std::string &filename)
{
    std::ifstream infile(filename);
    std::string line;
    std::vector<Example> result;
    Example cur;

    while (std::getline(infile, line))
    {
        if (line.length() == 0)
        {
            result.push_back(cur);
            cur = Example();
        }
        else
        {
            std::vector<std::string> vec;
            folly::split('\t', line, vec);
            cur.word.push_back(vec[1]);
            cur.pos.push_back(vec[4]);
            cur.head.push_back(folly::to<int>(vec[6]) - 1);
            cur.label.push_back(vec[7]);
        }
    }
    return result;
}

auto output_example(const Example &ex)
{
    std::cout << ex.head.size() << std::endl;
    for (auto i = 0; i < ex.head.size(); i++)
    {
        std::cout << i << " " << ex.word[i] << " " << ex.pos[i] << " " << ex.label[i] << " " << ex.head[i] << std::endl;
    }
}

enum class TransitionType
{
    left,
    right,
    shift
};

struct Transition
{
    TransitionType type;
    int label;
    bool operator==(const Transition &other) const
    {
        return label == other.label && type == other.type;
    }
};

namespace std
{
template <>
struct hash<Transition>
{
    std::size_t operator()(const Transition &k) const
    {
        return k.label << 2 + (int)k.type;
    }
};
}

std::vector<std::string> extract_unique(const std::vector<Example> &items, std::function<std::vector<std::string>(Example)> f)
{
    std::unordered_set<std::string> result;

    for (const auto &ex : items)
    {
        auto vals = f(ex);
        result.insert(begin(vals), end(vals));
    }

    return std::vector<std::string>(begin(result), end(result));
}

struct ParserData
{
    std::string root_label;
    EmbeddingData word_embeddings, pos_embeddings, label_embeddings;
    std::vector<Transition> id2tran;
    std::unordered_map<Transition, int> tran2id;

    ParserData(std::vector<Example> examples, std::string embeddings_filename, int embedding_size = 50);
    int tran2id2(const Transition& tran);
};

int ParserData::tran2id2(const Transition& tran) {
    if (tran.type == TransitionType::shift) {
        return 0;
    } else if (tran.type == TransitionType::left) {
        return tran.label + 1;
    } else {
        return tran.label + 1 + label_embeddings.id2tok.size();
    }

}

ParserData::ParserData(std::vector<Example> examples, std::string embeddings_filename, int embedding_size) : word_embeddings(extract_unique(examples, [](const auto &x) { return x.word; }), embedding_size),
                                                                                                             label_embeddings(extract_unique(examples, [](const auto &x) { return x.label; }), embedding_size),
                                                                                                             pos_embeddings(extract_unique(examples, [](const auto &x) { return x.pos; }), embedding_size),
                                                                                                             root_label("root")
{
    word_embeddings.load_from_file(embeddings_filename);
    int id = 0;

    auto addTran = [&id, this](Transition tran) {
        id2tran.push_back(tran);
        tran2id.insert({tran, id});

        id += 1;
    };

    addTran(Transition{TransitionType::shift, 0});
    for (int i = 0; i < label_embeddings.id2tok.size(); i++)
    {
        addTran(Transition{TransitionType::left, i});
        addTran(Transition{TransitionType::right, i});
    }
    std::cout << "Added " << id << " transition types" << std::endl;
}

struct Arc
{
    int head, dep, label;
};

struct ClassifierInstance
{
    std::vector<int> word_ids, pos_ids, label_ids;
    int trans_id;
};

struct Parser
{
    ParserData *data;
    std::vector<int> buf, stack; // indices into an example
    std::vector<Arc> arcs;

    Example *ex; // TODO: preprocess example to replace words/POS/etc with IDs right away
    bool step(Transition trans);
    Optional<ClassifierInstance> get_instance(bool ask_oracle);
    Optional<Transition> get_oracle();
    Parser(ParserData *pdata, Example *e);
};

Parser::Parser(ParserData *pdata, Example *e)
    : ex(e), data(pdata)
{
    // stack and arcs are initialized empty
    for (int i = 0; i < e->word.size(); i++)
    {
        buf.push_back(e->word.size() - i - 1);
    }
}

Optional<ClassifierInstance> Parser::get_instance(bool ask_oracle)
{
    ClassifierInstance result;
    if (ask_oracle)
    {
        auto oracle = get_oracle();
        if (!oracle) {
            return folly::none;
        }
        result.trans_id = data->tran2id[*oracle];
    }

    auto addFeatures = [&result, this](auto arr, int i) {
        if (i < arr.size())
        {
            int idx = arr[arr.size() - i - 1];
            auto word = ex->word[idx];
            result.word_ids.push_back(data->word_embeddings.get_embedding_id(word));
            auto pos = ex->pos[idx];
            result.pos_ids.push_back(data->pos_embeddings.get_embedding_id(pos));
        }
        else
        {
            result.word_ids.push_back(data->word_embeddings.NULL_ID);
            result.pos_ids.push_back(data->pos_embeddings.NULL_ID);
        }
    };

    for (int i = 0; i < 3; i++)
    {
        addFeatures(buf, i);
        addFeatures(stack, i);
    }

    return result;
    // TODO: add more features as in Chen&Manning
}

Optional<Transition> Parser::get_oracle()
{
    if (stack.size() < 2)
    { // nothing to arc, so have to shift
        return Transition{TransitionType::shift, 0};
    }

    auto i0 = stack[stack.size() - 1];
    auto i1 = stack[stack.size() - 2];

    auto h0 = ex->head[i0];
    auto h1 = ex->head[i1];

    auto l0 = data->label_embeddings.get_embedding_id(ex->label[i0]);
    auto l1 = data->label_embeddings.get_embedding_id(ex->label[i1]);

    if (i1 >= 0 && h1 == i0)
    {
        return Transition{TransitionType::left, l1};
    }
    else if (i1 >= 0 && h0 == i1)
    {
        if (std::find_if(begin(buf), end(buf), [i0, this](const auto &x) { return ex->head[x] == i0; }) == end(buf))
        {
            // no one ahead of us depends on i0, can pop it off stack
            return Transition{TransitionType::right, l0};
        }
        else
        {
            return Transition{TransitionType::shift, 0};
        }
    }
    else if (buf.size() > 0)
    {
        return Transition{TransitionType::shift, 0};
    }

    return folly::none;
}

bool Parser::step(Transition trans)
{
    if (trans.type == TransitionType::shift)
    {
        if (buf.size() == 0)
        {
            return false;
        }
        stack.push_back(buf[buf.size() - 1]);
        buf.pop_back();
    }
    else if (trans.type == TransitionType::left)
    {
        if (stack.size() < 2)
        {
            return false;
        }

        auto s1 = stack[stack.size() - 1];
        auto s2 = stack[stack.size() - 2];
        stack.pop_back();
        stack.pop_back();
        stack.push_back(s1);
        arcs.push_back(Arc{s1, s2, trans.label});
    }
    else
    {
        if (stack.size() < 2)
        {
            return false;
        }

        auto s1 = stack[stack.size() - 1];
        auto s2 = stack[stack.size() - 2];
        stack.pop_back();
        ;
        arcs.push_back(Arc{s2, s1, trans.label});
    }
    return true;
}

struct DataProvider
{
    ParserData *pdata;
    std::vector<Example> *examples;
    std::unique_ptr<Parser> parser;
    int example_id = 0;
    int step_id = 0;
    ClassifierInstance get_next()
    {
        while (true) {
        if (!parser) {
            parser = std::make_unique<Parser>(pdata, &(*examples)[example_id]);
        }
        if (step_id >= 2 * (*examples)[example_id].word.size()-1)
        {
            example_id = (example_id + 1) % examples->size();
            step_id = 0;
            parser = std::make_unique<Parser>(pdata, &(*examples)[example_id]);
        }

        step_id += 1;
        auto result = parser->get_instance(true);
        if (result) {
            parser->step(pdata->id2tran[result->trans_id]);
            return *result;
        }
        }
    }
};

void prep_FC_data(caffe2::NetDef &model, const std::string &suffix, int n_input, int n_output)
{
    // >>> W = init_net.UniformFill([], "W", shape=[1, 2], min=-1., max=1.)
    {
        auto op = model.add_op();
        op->set_type("UniformFill");
        auto arg1 = op->add_arg();
        arg1->set_name("shape");
        arg1->add_ints(n_output);
        arg1->add_ints(n_input);
        auto arg2 = op->add_arg();
        arg2->set_name("min");
        arg2->set_f(-1);
        auto arg3 = op->add_arg();
        arg3->set_name("max");
        arg3->set_f(1);
        op->add_output("W" + suffix);
    }

    // >>> B = init_net.ConstantFill([], "B", shape=[1], value=0.0)
    {
        auto op = model.add_op();
        op->set_type("ConstantFill");
        auto arg1 = op->add_arg();
        arg1->set_name("shape");
        arg1->add_ints(n_output);
        auto arg2 = op->add_arg();
        arg2->set_name("value");
        arg2->set_f(0);
        op->add_output("B" + suffix);
    }
}

void print(const caffe2::Blob *blob, const std::string &name)
{
    if (!blob)
    {
        std::cout << "NULL BLOB" << std::endl;
    }
    auto tensor = blob->Get<caffe2::TensorCPU>();
    const auto &data = tensor.data<float>();
    std::cout << name << "(" << tensor.dims()
              << "): " << std::vector<float>(data, data + tensor.size())
              << std::endl;
}

void prep_embedding(caffe2::Workspace& workspace, caffe2::NetDef& model, const std::string& input, EmbeddingData* data, const std::string output, 
    std::vector<caffe2::OperatorDef *>& gradient_ops) {
    const auto emb_name = input+"_embedding";
    const auto gathered_name = input + "_gathered";
    auto blob = workspace.CreateBlob(emb_name);
    auto tensor = workspace.GetBlob(emb_name)->GetMutable<caffe2::TensorCPU>();
    tensor->ResizeLike(data->tensor);
    tensor->ShareData(data->tensor);

       {
        auto op = model.add_op();
        op->set_type("Gather");
        op->add_input(emb_name);
        op->add_input(input);
        op->add_output(gathered_name);
//        auto arg1 = op->add_arg();
 //       arg1->set_name("dense_gradient");
  //      arg1->set_i(1);
 
        gradient_ops.push_back(op);
    } 

    {
        auto op = model.add_op();
        op->set_type("Flatten");
        op->add_input(gathered_name);
        op->add_output(output);
        gradient_ops.push_back(op);
    } 

}

void updateStep(caffe2::NetDef& predictModel, const std::string name) {
    auto op = predictModel.add_op();
    op->set_type("WeightedSum");
    op->add_input(name);
    op->add_input("ONE");
    op->add_input(name+"_grad");
    op->add_input("LR");
    op->add_output(name);
}

void model(caffe2::Workspace& workspace, ParserData* pdata, DataProvider* prov, int batch_size = 32)
{
    caffe2::NetDef initModel;
    initModel.set_name("init");



    {
        std::vector<int> data(batch_size);
        auto tensor = workspace.CreateBlob("label")->GetMutable<caffe2::TensorCPU>();
        auto value = caffe2::TensorCPU({batch_size}, data, NULL);
        tensor->ResizeLike(value);
        tensor->ShareData(value);
    }


     {
        std::vector<float> data(batch_size*6);
        auto tensor = workspace.CreateBlob("word_ids")->GetMutable<caffe2::TensorCPU>();
        auto value = caffe2::TensorCPU({batch_size, 6}, data, NULL);
        tensor->ResizeLike(value);
        tensor->ShareData(value);
    }

        {
        std::vector<float> data(batch_size*6);
        auto tensor = workspace.CreateBlob("pos_ids")->GetMutable<caffe2::TensorCPU>();
        auto value = caffe2::TensorCPU({batch_size, 6}, data, NULL);
        tensor->ResizeLike(value);
        tensor->ShareData(value);
    }

    caffe2::NetDef predictModel;
    predictModel.set_name("predict");

    const int n_features = 6 * 50 * 2;
    const int n_hidden = 200;
    const int n_output = pdata->id2tran.size();

    // >>> ITER = init_net.ConstantFill([], "ITER", shape=[1], value=0,
    // dtype=core.DataType.INT32)
    {
        auto op = initModel.add_op();
        op->set_type("ConstantFill");
        auto arg1 = op->add_arg();
        arg1->set_name("shape");
        arg1->add_ints(1);
        auto arg2 = op->add_arg();
        arg2->set_name("value");
        arg2->set_i(0);
        auto arg3 = op->add_arg();
        arg3->set_name("dtype");
        arg3->set_i(caffe2::TensorProto_DataType_INT64);
        op->add_output("ITER");
    }

    prep_FC_data(initModel, "_1", n_features, n_hidden);
    prep_FC_data(initModel, "_2", n_hidden, n_output);


    // store gradients
    std::vector<caffe2::OperatorDef *> gradient_ops;



    //workspace.CreateBlob("label_ids");

    // need to use Gather, then Flatten, then Concat

    prep_embedding(workspace, predictModel, "word_ids", &pdata->word_embeddings, "words_emb", gradient_ops);
    //prep_embedding(workspace, predictModel, "label_ids", &pdata->label_embeddings, "label_emb", gradient_ops);
    prep_embedding(workspace, predictModel, "pos_ids", &pdata->pos_embeddings, "pos_emb", gradient_ops);

    {
        auto op = predictModel.add_op();
        op->set_type("Concat");
        op->add_input("words_emb");
        //op->add_input("label_emb");
        op->add_input("pos_emb");
        op->add_output("data_gathered");
        op->add_output("__junk");
        gradient_ops.push_back(op);
    }

    {
        auto op = predictModel.add_op();
        op->set_type("FC");
        op->add_input("data_gathered");
        op->add_input("W_1");
        op->add_input("B_1");
        op->add_output("FC1");
        gradient_ops.push_back(op);
    }


    {
        auto op = predictModel.add_op();
        op->set_type("FC");
        op->add_input("data_gathered");
        op->add_input("W_1");
        op->add_input("B_1");
        op->add_output("FC1");
        gradient_ops.push_back(op);
    }

    {
        auto op = predictModel.add_op();
        op->set_type("Relu");
        op->add_input("FC1");
        op->add_output("FC1_relu");
        gradient_ops.push_back(op);
    }

    {
        auto op = predictModel.add_op();
        op->set_type("FC");
        op->add_input("FC1_relu");
        op->add_input("W_2");
        op->add_input("B_2");
        op->add_output("fc2_out");
        gradient_ops.push_back(op);
    }

    // // >>> pred = m.net.Sigmoid(fc_1, "pred")
    // {
    //     auto op = predictModel.add_op();
    //     op->set_type("Sigmoid");
    //     op->add_input("fc2_out");
    //     op->add_output("pred");
    //     gradient_ops.push_back(op);
    // }

    {
        auto op = predictModel.add_op();
        op->set_type("SoftmaxWithLoss");
        op->add_input("fc2_out");
        op->add_input("label");
        op->add_output("softmax");
        op->add_output("loss");
        gradient_ops.push_back(op);
    }

    // >>> m.AddGradientOperators([loss])
    {
        auto op = predictModel.add_op();
        op->set_type("ConstantFill");
        auto arg = op->add_arg();
        arg->set_name("value");
        arg->set_f(1.0);
        op->add_input("loss");
        op->add_output("loss_grad");
        op->set_is_gradient_op(true);
    }

  {
    auto op = initModel.add_op();
    op->set_type("ConstantFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    arg1->add_ints(1);
    auto arg2 = op->add_arg();
    arg2->set_name("value");
    arg2->set_f(1.0);
    op->add_output("ONE");
  }



  {
    auto op = predictModel.add_op();
    op->set_type("LearningRate");
    auto arg1 = op->add_arg();
    arg1->set_name("base_lr");
    arg1->set_f(-0.1);
    auto arg2 = op->add_arg();
    arg2->set_name("policy");
    arg2->set_s("step");
    auto arg3 = op->add_arg();
    arg3->set_name("stepsize");
    arg3->set_i(20);
    auto arg4 = op->add_arg();
    arg4->set_name("gamma");
    arg4->set_f(0.9);
    op->add_input("ITER");
    op->add_output("LR");
}

  {
    auto op = predictModel.add_op();
    op->set_type("Iter");
    op->add_input("ITER");
    op->add_output("ITER");
}

        std::cout << "test1" << std::endl;

    std::reverse(gradient_ops.begin(), gradient_ops.end());
    for (auto op : gradient_ops)
    {
        std::vector<caffe2::GradientWrapper> output(op->output_size());
        for (auto i = 0; i < output.size(); i++)
        {
            output[i].dense_ = op->output(i) + "_grad";
        }
        caffe2::GradientOpsMeta meta = GetGradientForOp(*op, output);
        if (meta.ops_.size() == 0) {
            std::cout << output.size() << " " << op->type() << " " << meta.g_input_.size() << std::endl;
            continue;
            //will fail here
        }
        auto grad = predictModel.add_op();
        grad->CopyFrom(meta.ops_[0]);
        grad->set_is_gradient_op(true);
    }

    std::cout << "test2" << std::endl;
    // we now have a model

    updateStep(predictModel, "W_1");
    updateStep(predictModel, "W_2");
    updateStep(predictModel, "B_1");
    updateStep(predictModel, "B_2");
    //updateStep(predictModel, "word_ids_embedding");
    //updateStep(predictModel, "pos_ids_embedding");
    


    CAFFE_ENFORCE(workspace.RunNetOnce(initModel));

    std::cout << "Init run" << std::endl;
    CAFFE_ENFORCE(workspace.CreateNet(predictModel));

    std::cout << "created" << std::endl;
    // let's finally do training now

    for (int i=0; i < 500; i++) {
        
        // collect training_data
        std::vector<int> word_ids;
        std::vector<int> pos_ids;
        std::vector<int> labels;
        for (int j=0; j<batch_size; j++) {
            auto instance = prov->get_next();
            labels.push_back(instance.trans_id);
            word_ids.insert(end(word_ids), begin(instance.word_ids), end(instance.word_ids));
            pos_ids.insert(end(pos_ids), begin(instance.pos_ids), end(instance.pos_ids));
        }


        {
        auto tensor = workspace.GetBlob("label")->GetMutable<caffe2::TensorCPU>();
        auto value = caffe2::TensorCPU({batch_size}, labels, NULL);
        tensor->ShareData(value);
        }

         {
        auto tensor = workspace.GetBlob("word_ids")->GetMutable<caffe2::TensorCPU>();
        auto value = caffe2::TensorCPU({batch_size, 6}, word_ids, NULL);
        tensor->ShareData(value);
        }
 
        {
        auto tensor = workspace.GetBlob("pos_ids")->GetMutable<caffe2::TensorCPU>();
        auto value = caffe2::TensorCPU({batch_size, 6}, pos_ids, NULL);
        tensor->ShareData(value);
        }


        CAFFE_ENFORCE(workspace.RunNet(predictModel.name()));
        std::cout << "step: " << i << ": ";
        print(workspace.GetBlob("loss"), "loss");
    }
}

 

int run()
{
    caffe2::Workspace workspace;
    auto res = read_conll("data/train.conll");
    std::cout << res.size() << std::endl;
    output_example(res[22]);

    ParserData pdata{res, "data/en-cw.txt", 50};
    // Parser p{&pdata, &res[22]};
    // for (int i = 0; i < 2 * res[22].word.size(); i++)
    // {
    //     auto trans = p.get_oracle();
    //     p.step(*trans);
    // }

    // std::cout << "Arcs: " << p.arcs.size() << std::endl;
    // for (const auto &arc : p.arcs)
    // {
    //     std::cout << p.ex->word[arc.head] << "->" << p.ex->word[arc.dep] << std::endl;
    // }
    // std::cout << "Stack" << p.stack << " Buf: " << p.buf << std::endl;


    DataProvider dprov{&pdata, &res};
    model(workspace, &pdata, &dprov);

    return 0;
}

int main(int argc, char **argv)
{
    caffe2::GlobalInit(&argc, &argv);
    run();
    //google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
