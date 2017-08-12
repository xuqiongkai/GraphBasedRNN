
local IterLogisticRegression = torch.class('cora.IterLogisticRegression')


function IterLogisticRegression:__init(config)
    self.learning_rate  = config.learning_rate  or 0.05
    self.reg            = config.reg            or 1e-4
    self.iter_num       = config.iter_num       or 5
    self.feature_num    = config.feature_num
    self.label_num      = config.label_num
    self.sample_num     = config.sample_num
    self.batch_size     = config.batch_size

    self.model_structure= config.model_structure or 'ica_c' -- 'ica_c' or 'ica_b'
    self.module         = self:new_lr_module()
    self.optim_state = { learningRate = self.learning_rate }

    self.criterion      = nn.ClassNLLCriterion()
    self.params, self.grad_params = self.module:getParameters()
end

function IterLogisticRegression:new_lr_module()
    local input = nn.Identity()()
    local hidden = nn.Linear(self.feature_num + self.label_num, self.label_num){input}
    local output = nn.LogSoftMax()(hidden)
    local module = nn.gModule({input}, {output})
    return module
end

function IterLogisticRegression:train(features, labels, cites, indices, train_indices_set, i2s, s2i, label_map)
    local total_loss = 0
    local train_num = indices:size()[1]
    for i = 1, train_num, self.batch_size do
        xlua.progress(i, train_num)
        local batch_size = math.min(i + self.batch_size - 1, train_num) - i + 1
        self.grad_params:zero()


        local feval = function(x)
        local batch_loss = 0
        local it
        for j = 1, batch_size do
            local idx = indices[i + j - 1]
            local label = label_map[labels[i2s[idx]]]
            local feature = features[idx]
            local feature_near = torch.zeros(self.label_num)
            local near_idx = cites[i2s[idx]]
            if near_idx ~= nil then
                for _, v in ipairs(near_idx) do
                    if train_indices_set[s2i[v]] == true then
                        it = label_map[labels[v]]
                        if self.model_structure == 'ica_c' then
                            feature_near[it] = feature_near[it] + 1
                        elseif self.model == 'ica_b' then
                            feature_near[it] = 1
                        end
                    end

                end
            end
            feature = torch.cat(feature, feature_near, 1)
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

function IterLogisticRegression:predict(features, predictions, cites, i2s, s2i, label_map)
    local test_num = #i2s
    local new_predictions = {}
    local p
    for i = 1, test_num do
        local idx = i
        local feature = features[idx]
        local feature_near = torch.zeros(self.label_num)
        local near_idx = cites[i2s[idx]]
        if predictions ~= nil and near_idx ~= nil then
            for _, v in ipairs(near_idx) do
                p = predictions[v]
                if p ~= nil then
                    if self.model_structure == 'ica_c' then
                        feature_near[p] = feature_near[p] + 1
                    elseif self.model_structure == 'ica_b' then
                        feature_near[p] = 1
                    end
                end
            end
        end
        feature = torch.cat(feature, feature_near, 1)
        local output = self.module:forward(feature)
        new_predictions[i2s[i]] = util.best_label(output, self.label_num)
    end
    return new_predictions
end

function IterLogisticRegression:test(features, labels, cites, indices, test_indices_set, i2s, s2i, label_map)
    local result = {}
    local test_num = indices:size()[1]
    local predictions = torch.Tensor(test_num)
    local golds = torch.Tensor(test_num)
    local all_predictions
    for i = 1, self.iter_num do
        printf("iter: %d\n", i)
        all_predictions = self:predict(features, all_predictions, cites, i2s, s2i, label_map)
    end

    for i = 1, test_num do
        local idx = indices[i]
        predictions[i] = all_predictions[i2s[idx]]
        golds[i] = label_map[labels[i2s[idx]]]
    end

    result.acc = eval.accuracy(predictions, golds)
    return result
end


