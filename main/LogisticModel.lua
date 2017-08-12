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

local LogisticRegression = torch.class('cora.LogisticRegression')


function LogisticRegression:__init(config)
    self.learning_rate  = config.learning_rate  or 0.1
    self.reg            = config.reg            or 1e-4
    self.feature_num    = config.feature_num
    self.label_num      = config.label_num
    self.sample_num     = config.sample_num
    self.batch_size     = config.batch_size
    self.optim_state = { learningRate = self.learning_rate }

    self.task_type      = config.task_type      or 'classification' -- 'regression' or 'classification'
    if self.task_type == 'classification' then
        self.module         = self:new_lr_module_classification()
        self.criterion      = nn.ClassNLLCriterion()
    elseif self.task_type == 'regression' then
        self.module         = self:new_lr_module_regression()
        self.criterion      = nn.MSECriterion()
    end

    self.params, self.grad_params = self.module:getParameters()
end

function LogisticRegression:show()

end

function LogisticRegression:new_lr_module_classification()
    local input = nn.Identity()()
    local hidden = nn.Linear(self.feature_num, self.label_num){input}
    local output= nn.LogSoftMax()(hidden)
    local module = nn.gModule({input}, {output})
    return module
end

function LogisticRegression:new_lr_module_regression()
    local input = nn.Identity()()
    local hidden = nn.Linear(self.feature_num, 1){input}
    local module = nn.gModule({input}, {hidden})
    return module
end


function LogisticRegression:train(features, labels, cites, indices, train_indices_set, i2s, s2i, label_map)
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
            local label
            if self.task_type == 'classification' then
                label = label_map[labels[i2s[idx]]]
            elseif self.task_type == 'regression' then
                label = labels[i2s[idx]]
            end

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


function LogisticRegression:test(features, labels, cites, indices, test_indices_set, i2s, s2i, label_map)
    local result = {}
    local test_num = indices:size()[1]
    local predictions = torch.Tensor(test_num)
    local golds = torch.Tensor(test_num)
    local tmp = 0
    for i = 1, test_num do
        if i % 50 == 0 then xlua.progress(i, test_num) end
        local idx = indices[i]
        local feature = features[idx]
        local output = self.module:forward(feature)
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


