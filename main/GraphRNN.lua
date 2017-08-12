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


local GraphRNN = torch.class('cora.GraphRNN')

function GraphRNN:__init(config)
    self.learning_rate  = config.learning_rate  or 0.1

    self.reg            = config.reg            or 1e-4
    self.batch_size     = config.batch_size     or 25
    self.feature_num    = config.feature_num
    self.mem_dim        = config.mem_dim        or 200
    self.label_num      = config.label_num
    self.tree_depth     = config.tree_depth     or 2

    self.task_type      = config.task_type      or 'classification' -- 'regression' or 'classification'
    if self.task_type == 'classification' then
        self.criterion      = nn.ClassNLLCriterion()
    elseif self.task_type == 'regression' then
        self.criterion      = nn.MSECriterion()
    end
    self.attention      = nil
    self.type           = "lstm"
    if config.model_structure == "lstm" then
        self.type = "lstm"
    elseif config.model_structure == "rnn" then
        self.type = "rnn"
    elseif config.model_structure == "lstm_att_tar" then
        self.attention = 'tar'
        self.type = "lstm"
    elseif config.model_structure == "lstm_att_par" then
        self.attention = 'par'
        self.type = "lstm"
    elseif config.model_structure == "rnn_att_tar" then
        self.attention = 'tar'
        self.type = "rnn"
    elseif config.model_structure == "rnn_att_par" then
        self.attention = 'par'
        self.type = "rnn"
    end

    local treelstm_config = {
        in_dim  = self.feature_num,
        mem_dim = self.mem_dim,
        attention = self.attention
    }

    treelstm_config.output_module_fn = function() return self:new_output_module() end
    treelstm_config.criterion = self.criterion

    if self.type == "lstm" then
        if self.attention ~= nil then
            self.tree_module = classifier.AttentionChildSumTreeLSTM(treelstm_config)
        else
            self.tree_module = classifier.ChildSumTreeLSTM(treelstm_config)
        end
    elseif self.type == "rnn" then
        if self.attention ~= nil then
            self.tree_module = classifier.ChildSumTreeRNN(treelstm_config)
        else
            self.tree_module = classifier.ChildSumTreeRNN(treelstm_config)
        end
    end

    self.output_module = self:new_output_module()


    local modules = nn.Parallel()
    :add(self.tree_module)
    :add(self.output_module)
    self.params, self.grad_params = modules:getParameters()


    self.optim_state = { learningRate = self.learning_rate }

end

function GraphRNN:new_output_module()
    local rep = nn.Identity()()
    local hidden, output
    if self.task_type == 'classification' then
        hidden = nn.Linear(self.mem_dim, self.label_num)(rep)
        output = nn.LogSoftMax()(hidden)
    elseif self.task_type == 'regression' then
        output = nn.Linear(self.mem_dim, 1)(rep)
    end
    local module = nn.gModule({rep}, {output})
    return module
end

function GraphRNN:generate_tree(idx, depth, train_indices_set, labels, cites, i2s, s2i, label_map)
    local queue = {}
    local head = 0
    local head_node

    -- first root node
    local root = util.Tree()
    root.depth = 0
    root.idx = idx
--    if train_indices_set[idx] == true then
--        root.gold_label = label_map[labels[i2s[idx]]]
--    end

    queue[1] = root

    local i, ns, node
    while head < #queue do
        head = head + 1
        head_node = queue[head]
        ns = cites[i2s[head_node.idx]]

        if ns ~= nil and head_node.depth < depth then
            local count = 0
            for _, v in ipairs(ns) do
                if count > 6 then break end
                count = count + 1

                i = s2i[v]
                if i ~= nil then
                node = util.Tree()
                node.depth = head_node.depth + 1
                node.idx = i
--                if train_indices_set[i] == true then
--                    node.gold_label = label_map[labels[v]]
--                end
                queue[#queue + 1] = node
                head_node:add_child(node)
                end

            end

        end
    end
--    print(root:depth_first_preorder())
    return root, #queue
end


function GraphRNN:train(features, labels, cites, indices, train_indices_set, i2s, s2i, label_map, reset_att_wight)
    local total_loss = 0
    local train_num = indices:size()[1]
    local zeros = torch.zeros(self.mem_dim)
    self.tree_module.train = true
    local batch_norm = 0
    for i = 1, train_num, self.batch_size do
        xlua.progress(i, train_num)
        local batch_size = math.min(i + self.batch_size - 1, train_num) - i + 1
    local feval = function(x)
        self.grad_params:zero()

        local batch_loss = 0
        batch_norm = 0
        for j = 1, batch_size do
            local idx = indices[i + j - 1]
            local label
            if self.task_type == 'classification' then
                label = label_map[labels[i2s[idx]]]
            elseif self.task_type == 'regression' then
                label = labels[i2s[idx]]
            end

            local tree, tree_size = self:generate_tree(idx, self.tree_depth, train_indices_set, labels, cites, i2s, s2i, label_map)
            batch_norm = batch_norm + tree_size
            local rep

            if self.type == "lstm" then
                if self.attention ~= nil then
                    rep = self.tree_module:forward(tree, tree.idx, features)[2]
                else
                    rep = self.tree_module:forward(tree, features)[2]
                end
            elseif self.type == "rnn" then
                if self.attention ~= nil then
                    rep = self.tree_module:forward(tree, features)
                else
                    rep = self.tree_module:forward(tree, features)
                end
            end
            local output = self.output_module:forward(rep)
            batch_loss = batch_loss + self.criterion:forward(output, label)

            local output_grad = self.criterion:backward(output, label)
            local rep_grad = self.output_module:backward(rep, output_grad)
            local input_grads

            if self.type == "lstm" then
                if self.attention ~= nil then
                    input_grads = self.tree_module:backward(tree, tree.idx, features, {zeros, rep_grad})
                else
                    input_grads = self.tree_module:backward(tree, features, {zeros, rep_grad})
                end
            elseif self.type == "rnn" then
                if self.attention ~= nil then
                    input_grads = self.tree_module:backward(tree, features, rep_grad)
                else
                    input_grads = self.tree_module:backward(tree, features, rep_grad)
                end
            end
        end

        total_loss = total_loss + batch_loss
        batch_loss = batch_loss / batch_size
        return batch_loss, self.grad_params
    end
        self.grad_params:div(batch_norm)

        optim.adagrad(feval, self.params, self.optim_state)

    end
    xlua.progress(train_num, train_num)
    self.tree_module.train = false

end

function GraphRNN:test(features, labels, cites, indices, test_indices_set, i2s, s2i, label_map)
    local result = {}
    local test_num = indices:size()[1]
    local predictions = torch.Tensor(test_num)
    local golds = torch.Tensor(test_num)
    local tmp = 0
    for i = 1, test_num do
        if i % 50 == 0 then xlua.progress(i, test_num) end
        local idx = indices[i]
        local tree = self:generate_tree(idx, self.tree_depth, test_indices_set, labels, cites, i2s, s2i, label_map)
        local rep
        if self.type == "lstm" then
            if self.attention ~= nil then
                rep = self.tree_module:forward(tree, tree.idx,  features)[2]
            else
                rep = self.tree_module:forward(tree, features)[2]
            end
        elseif self.type == "rnn" then
            if self.attention ~= nil then
                rep = self.tree_module:forward(tree, features)
            else
                rep = self.tree_module:forward(tree, features)
            end
        end
        local output = self.output_module:forward(rep)

        if self.task_type == 'classification' then
            predictions[i] = util.best_label(output, self.label_num)
            golds[i] = label_map[labels[i2s[idx]]]
        elseif self.task_type == 'regression' then
            local label = labels[i2s[idx]]
            local delta = output - label
            tmp = tmp + delta * delta
        end

    end
    xlua.progress(test_num, test_num)
    if self.task_type == 'classification' then
        result.acc = eval.accuracy(predictions, golds)
    elseif self.task_type == 'regression' then
        result.acc = 0.5*tmp/test_num
    end
    return result
end


function GraphRNN:print(features, labels, cites, indices, test_indices_set, i2s, s2i, label_map)
    local test_num = indices:size()[1]
    for i = 1, test_num do
        local idx = indices[i]
        local tree = self:generate_tree(idx, self.tree_depth, test_indices_set, labels, cites, i2s, s2i, label_map)
        local rep
        print(i)
        self.tree_module:getWeight(tree, tree.idx, features, labels, i2s)
        self.tree_module:print(tree, 0, 1)
    end
end
