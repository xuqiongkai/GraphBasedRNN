--------------------------------------------------------------------------------
--
--     Graph-Based Recursive Neural Network for Vertex Classification
--     Copyright (C) 2016-2017  Qiongkai Xu, Chenchen Xu
--
--     Copyright (c) 2016  Kai Sheng Tai, Richard Socher, 
--                         and Christopher Manning
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

--[[

  A Child-Sum Tree-RNN with input at each node.

--]]

local ChildSumTreeRNN, parent = torch.class('classifier.ChildSumTreeRNN', 'classifier.TreeLSTM')

function ChildSumTreeRNN:__init(config)
  parent.__init(self, config)
  self.gate_output = config.gate_output or true

  -- a function that instantiates an output module that takes the hidden state h as input
  self.output_module_fn = config.output_module_fn
  self.criterion = config.criterion

  -- composition module
  self.composer = self:new_composer()
  self.composers = {}

  -- output module
  self.output_module = self:new_output_module()
  self.output_modules = {}
end

--function ChildSumTreeRNN:new_composer()
--  local input = nn.Identity()()
--  local child_c = nn.Identity()()
--  local child_h = nn.Identity()()
--  local child_h_sum = nn.Mean(1)(child_h)
----Mean
--  local i = nn.Sigmoid()(
--    nn.CAddTable(){
--      nn.Linear(self.in_dim, self.mem_dim)(input),
--      nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum)
--    })
--  local f = nn.Sigmoid()(
--    classifier.CRowAddTable(){
--      nn.TemporalConvolution(self.mem_dim, self.mem_dim, 1)(child_h),
--      nn.Linear(self.in_dim, self.mem_dim)(input),
--    })
--  local update = nn.Tanh()(
--    nn.CAddTable(){
--      nn.Linear(self.in_dim, self.mem_dim)(input),
--      nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum)
--    })
--  local c = nn.CAddTable(){
--      nn.CMulTable(){i, update},
--      nn.Sum(1)(nn.CMulTable(){f, child_c})
--    }
--
--  local h
--  if self.gate_output then
--    local o = nn.Sigmoid()(
--      nn.CAddTable(){
--        nn.Linear(self.in_dim, self.mem_dim)(input),
--        nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum)
--      })
--    h = nn.CMulTable(){o, nn.Tanh()(c)}
--  else
--    h = nn.Tanh()(c)
--  end
--
--  local composer = nn.gModule({input, child_c, child_h}, {c, h})
--  if self.composer ~= nil then
--    share_params(composer, self.composer)
--  end
--  return composer
--end

function ChildSumTreeRNN:new_composer()
  local input = nn.Identity()()
  local child_h = nn.Identity()()
  local child_h_pool = nn.Max(1)(child_h)

  local h = nn.Tanh()(
    nn.CAddTable(){
      nn.Linear(self.in_dim, self.mem_dim)(input),
--      nn.Sum(1)(nn.TemporalConvolution(self.mem_dim, self.mem_dim, 1)(child_h))
      nn.Linear(self.mem_dim, self.mem_dim)(child_h_pool)
    })

  local composer = nn.gModule({input, child_h}, {h})
  if self.composer ~= nil then
    share_params(composer, self.composer)
  end
  return composer
end

function ChildSumTreeRNN:new_output_module()
  if self.output_module_fn == nil then return nil end
  local output_module = self.output_module_fn()
  if self.output_module ~= nil then
    share_params(output_module, self.output_module)
  end
  return output_module
end

function ChildSumTreeRNN:forward(tree, inputs)
  local loss = 0
  for i = 1, tree.num_children do
    local _, child_loss = self:forward(tree.children[i], inputs)
    loss = loss + child_loss
  end
  local child_h = self:get_child_states(tree)
  self:allocate_module(tree, 'composer')
  tree.state = tree.composer:forward{inputs[tree.idx], child_h}

  if self.output_module ~= nil then
    self:allocate_module(tree, 'output_module')
    tree.output = tree.output_module:forward(tree.state)
    if self.train and tree.gold_label ~= nil then
      loss = loss + self.criterion:forward(tree.output, tree.gold_label)
    end
  end
  return tree.state, loss
end

function ChildSumTreeRNN:backward(tree, inputs, grad)
  local grad_inputs = torch.Tensor(inputs:size())
  self:_backward(tree, inputs, grad, grad_inputs)
  return grad_inputs
end

function ChildSumTreeRNN:_backward(tree, inputs, grad, grad_inputs)
  local output_grad = self.mem_zeros
  if tree.output ~= nil and tree.gold_label ~= nil then
    output_grad = tree.output_module:backward(
      tree.state[2], self.criterion:backward(tree.output, tree.gold_label))
  end
  self:free_module(tree, 'output_module')
  tree.output = nil

  local child_h = self:get_child_states(tree)
  local composer_grad = tree.composer:backward(
    {inputs[tree.idx], child_h},
    grad + output_grad)
  self:free_module(tree, 'composer')
  tree.state = nil

  grad_inputs[tree.idx] = composer_grad[1]
  local child_h_grads = composer_grad[2]
  for i = 1, tree.num_children do
    self:_backward(tree.children[i], inputs, child_h_grads[i], grad_inputs)
  end
end

function ChildSumTreeRNN:clean(tree)
  self:free_module(tree, 'composer')
  self:free_module(tree, 'output_module')
  tree.state = nil
  tree.output = nil
  for i = 1, tree.num_children do
    self:clean(tree.children[i])
  end
end

function ChildSumTreeRNN:parameters()
  local params, grad_params = {}, {}
  local cp, cg = self.composer:parameters()
  tablex.insertvalues(params, cp)
  tablex.insertvalues(grad_params, cg)
  if self.output_module ~= nil then
    local op, og = self.output_module:parameters()
    tablex.insertvalues(params, op)
    tablex.insertvalues(grad_params, og)
  end
  return params, grad_params
end

function ChildSumTreeRNN:get_child_states(tree)
  local child_h
  if tree.num_children == 0 then
    child_h = torch.zeros(1, self.mem_dim)
  else
    child_h = torch.Tensor(tree.num_children, self.mem_dim)
    for i = 1, tree.num_children do
       child_h[i] = tree.children[i].state
    end
  end
  return child_h
end
