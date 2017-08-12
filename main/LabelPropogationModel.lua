
local LabelPropagation = torch.class('cora.LabelPropagation')


function LabelPropagation:__init(config)
    self.learning_rate  = config.learning_rate  or 0.1
    self.reg            = config.reg            or 1e-4
    self.feature_num    = config.feature_num
    self.label_num      = config.label_num
    self.sample_num     = config.sample_num
    self.batch_size     = config.batch_size
    self.module         = self:new_lr_module()
    self.optim_state = { learningRate = self.learning_rate }
    self.propagate_rate = config.propagate_rate or 0.1
    self.iter_round     = config.iter_round     or 50

    self.criterion      = nn.ClassNLLCriterion()
    self.params, self.grad_params = self.module:getParameters()
end


function LabelPropagation:new_lr_module()
    local input = nn.Identity()()
    local hidden = nn.Linear(self.feature_num, self.label_num){input}
    local output = nn.LogSoftMax()(hidden)
    local module = nn.gModule({input}, {output})
    return module
end

function LabelPropagation:train(features, labels, cites, indices, train_indices_set, i2s, s2i, label_map)
    local total_loss = 0
    local train_num = indices:size()[1]
    for i = 1, train_num, self.batch_size do
        xlua.progress(i, train_num)
        local batch_size = math.min(i + self.batch_size - 1, train_num) - i + 1
        self.grad_params:zero()


        local feval = function(x)
        local batch_loss = 0
        for j = 1, batch_size do
            local idx = indices[i + j - 1]
            local label = label_map[labels[i2s[idx]]]
            local feature = features[idx]
            local output = self.module:forward(feature)
            batch_loss = batch_loss + self.criterion:forward(output, label)
            local output_grad = self.criterion:backward(output, label)
            local feature_grad = self.module:backward(feature, output_grad)
        end
        total_loss = total_loss + batch_loss
        batch_loss = batch_loss / batch_size
        self.grad_params:div(batch_size)
        return batch_loss, self.grad_params
    end
    optim.adagrad(feval, self.params, self.optim_state)

    end
    xlua.progress(train_num, train_num)
    total_loss = total_loss / train_num
    print('Train loss: '..total_loss)
end


function LabelPropagation:test(features, labels, cites, indices, test_indices_set, i2s, s2i, label_map)
    local result = {}
    local test_num = indices:size()[1]
    local all_num = features:size()[1]
    local init_prob = torch.Tensor(all_num, self.label_num)
    local iter_prob, spread_prob
    local predictions = torch.Tensor(test_num)
    local golds = torch.Tensor(test_num)
    -- initial predict
    for i = 1, test_num do
        if i % 50 == 0 then xlua.progress(i, test_num) end
        local idx = indices[i]
        golds[i] = label_map[labels[i2s[idx]]]
    end
    for idx = 1, all_num do
        local feature = features[idx]
        local output = self.module:forward(feature)
        init_prob[idx] = torch.exp(output)
    end
    xlua.progress(test_num, test_num)

    -- propogate
    iter_prob = init_prob
    local neighbors, n_idx
    for _ = 1, 20 do
        spread_prob = torch.zeros(all_num, self.label_num)
        for idx = 1, all_num do
            neighbors = cites[i2s[idx]]
            if neighbors ~= nil then
            for _, v in ipairs(neighbors) do
                n_idx = s2i[v]
                if n_idx ~= nil then
                    spread_prob[n_idx] = spread_prob[n_idx] + iter_prob[idx] / #neighbors
                end
            end
            end
        end
        iter_prob = iter_prob * (1-self.propagate_rate) + spread_prob * self.propagate_rate
    end

    -- assign label and evaluate
    for i = 1, test_num do
        local idx = indices[i]
        predictions[i] = util.best_label(iter_prob[idx], self.label_num)
    end
    result.acc = eval.accuracy(predictions, golds)
    return result
end


