--------------------------------------------------------------------------------
--
--     Graph-Based Recursive Neural Network for Vertex Classification
--     Copyright (C) 2016-2017  Qiongkai Xu, Chenchen Xu
--
--     This program is free software: you can redistribute it and/or modify
--     it under the terms of the GNU General Public License as published by
--     the Free Software Foundation, either version 3 of the License, or
--     (at your option) any later version.
--
--     This program is distributed in the hope that it will be useful,
--     but WITHOUT ANY WARRANTY; without even the implied warranty of
--     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
--     GNU General Public License for more details.
--
--     You should have received a copy of the GNU General Public License
--     along with this program.  If not, see <http://www.gnu.org/licenses/>.
--
--------------------------------------------------------------------------------


require '..'


local args = lapp [[
  -m,--model  (default lr)           Model architecture: lr(logistic regression),ica(_count, _binary),lstm, rnn]
  -d,--dataset (default cora)        Dataset: cora,citeseer,webkb,webkb-syn
  -p,--partition (default 0.8)       Partition: percentage for training data
  -l,--learning_rate (default 0.1)   Learning rate
  -e,--epochs (default 10)           Number of training epochs
  -h,--hidden_dim    (default 200)   LSTM/RNN hidden state dimension
  -s,--steps    (default 2)          LSTM/RNN steps to spread local tree
  -b,--batch_size (default 50)       Batch size
  -r,--random_seed (default 100)     Random seed
  -w,--step_wise (default 0)         Use stepwise training or not for attentive grnn model
]]



torch.manualSeed(args.random_seed)
local epoch_num = args.epochs
local content_path, cites_path, label_path, meta_path, feature_path
local dataset_path = "./dataset"


if args.dataset == "cora" then
    content_path = dataset_path.."/cora/cora.content"
    cites_path = dataset_path.."/cora/cora.cites"
    label_path = dataset_path.."/cora/cora.label"
    meta_path = dataset_path.."/cora/cora.meta"
    feature_path = dataset_path.."/cora/cora.feature"
elseif args.dataset == "citeseer" then
    cites_path = dataset_path.."/citeseer/citeseer.cites"
    label_path = dataset_path.."/citeseer/citeseer.label"
    meta_path = dataset_path.."/citeseer/citeseer.meta"
    feature_path = dataset_path.."/citeseer/citeseer.feature.1000"
elseif args.dataset == "webkb" or args.dataset == "webkb-syn" then
    if args.dataset == "webkb" then
        cites_path = dataset_path.."/WebKB/WebKB.cites"
    else
        cites_path = dataset_path.."/WebKB/WebKB.sim.cites"
    end
    label_path = dataset_path.."/WebKB/WebKB.label"
    meta_path = dataset_path.."/WebKB/WebKB.meta"
    feature_path = dataset_path.."/WebKB/WebKB.feature"
end
local noise_suffix=""
cites_path = cites_path..noise_suffix

printf("read data from: \n  %s\n  %s\n", cites_path, feature_path)

local features = util.read_features(feature_path)
local i2s, s2i, labels = util.read_labels(label_path)
local label_map, label_num = util.read_meta(meta_path)
local cites = util.read_cites(cites_path)
local sample_num, feature_num = features:size()[1], features:size()[2]


if args.model == 'lr' then
    model_class = cora.LogisticRegression
elseif args.model == 'ica_b' or args.model == "ica_c" then
    model_class = cora.IterLogisticRegression
elseif args.model == 'lp' then
    model_class = cora.LabelPropagation
elseif args.model == 'lstm' or args.model == 'lstm_att_tar' or args.model == 'lstm_att_par'then
    model_class = cora.GraphRNN
elseif args.model == 'rnn' or args.model == 'rnn_att_tar' or args.model == 'rnn_att_par'then
    model_class = cora.GraphRNN
end

model = model_class {
    feature_num = feature_num,
    label_num = label_num,
    sample_num = sample_num,
    batch_size = args.batch_size,
    learning_rate = args.learning_rate,
    model_structure = args.model,
    tree_depth = args.steps,
    step_wise = args.step_wise
}
local train_indices, test_indices, train_indices_set, test_indices_set = util.generate_indices(sample_num, args.partition)

local best_score = -1.0
for i = 1, epoch_num do
    local start = sys.clock()
    printf('-- epoch %d\n', i)
    if args.step_wise ~= 0 and i == 1 then
        model.tree_module:refresh_attention_weight('r')
        model:train(features, labels, cites, train_indices, train_indices_set, i2s, s2i, label_map, false)
    else
        model:train(features, labels, cites, train_indices, train_indices_set, i2s, s2i, label_map, false)
    end
    printf('-- finished epoch in %.2fs\n', sys.clock() - start)
    local result = model:test(features, labels, cites, test_indices, test_indices_set, i2s, s2i, label_map)
    printf('-- test score: %.6f\n', result.acc)
    if result.acc >= best_score then
            best_score = result.acc
    end

end
printf('-- best score: %.6f\n', best_score)




