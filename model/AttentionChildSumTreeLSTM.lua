--[[

  A Child-Sum Tree-LSTM with input at each node.

--]]

local AttentionChildSumTreeLSTM, parent = torch.class('classifier.AttentionChildSumTreeLSTM', 'classifier.TreeLSTM')

function AttentionChildSumTreeLSTM:__init(config)
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

  self.attention = config.attention
  print(self.attention)
end
function AttentionChildSumTreeLSTM:new_composer()
  local input = nn.Identity()()
  local child_c = nn.Identity()()
  local child_h = nn.Identity()()
  local input_t = nn.Identity()()
  local child_hw = nn.TemporalConvolution(self.mem_dim, self.in_dim, 1)(child_h)
  local twh = nn.SoftMax()(nn.MM(false, true){nn.Replicate(1,1)(input_t), child_hw })
  local child_h_pool = nn.MM(){twh, child_h}

  local i = nn.Sigmoid()(
    nn.CAddTable(){
      nn.Linear(self.in_dim, self.mem_dim)(input),
      nn.Linear(self.mem_dim, self.mem_dim)(child_h_pool)
    })
  local f = nn.Sigmoid()(
    classifier.CRowAddTable(){
      nn.TemporalConvolution(self.mem_dim, self.mem_dim, 1)(child_h),
      nn.Linear(self.in_dim, self.mem_dim)(input),
    })
  local update = nn.Tanh()(
    nn.CAddTable(){
      nn.Linear(self.in_dim, self.mem_dim)(input),
      nn.Linear(self.mem_dim, self.mem_dim)(child_h_pool)
    })
  local c = nn.CAddTable(){
      nn.CMulTable(){i, update},
      nn.Sum(1)(nn.CMulTable(){f, child_c})
    }

  local h

  if self.gate_output then
    local o = nn.Sigmoid()(
      nn.CAddTable(){
        nn.Linear(self.in_dim, self.mem_dim)(input),
        nn.Linear(self.mem_dim, self.mem_dim)(child_h_pool)
      })
    h = nn.CMulTable(){o, nn.Tanh()(c)}
  else
    h = nn.Tanh()(c)
  end

  local composer = nn.gModule({input, child_c, child_h, input_t}, {c, h})
  if self.composer ~= nil then
    share_params(composer, self.composer)
  end
  return composer
end


function AttentionChildSumTreeLSTM:refresh_attention_weight(type)
--  for indexNode, node in ipairs(self.composer.forwardnodes) do
--    if node.data.module then
--      print(indexNode, node.data.module)
--    end
--  end

print(type)
  if type =='r' then
    self.composer.forwardnodes[11].data.module.weight = torch.randn(self.in_dim, self.mem_dim) * 0.001
    print(self.composer.forwardnodes[11].data.module.weight)
  else
    self.composer.forwardnodes[11].data.module.weight = torch.zeros(self.in_dim, self.mem_dim)
  end
--  print(self.composer.forwardnodes[11].data.module.weight)
end


function AttentionChildSumTreeLSTM:new_output_module()
  if self.output_module_fn == nil then return nil end
  local output_module = self.output_module_fn()
  if self.output_module ~= nil then
    share_params(output_module, self.output_module)
  end
  return output_module
end

function AttentionChildSumTreeLSTM:forward(tree, idx_t, inputs)
  local loss = 0
  for i = 1, tree.num_children do
    local _, child_loss = self:forward(tree.children[i], idx_t, inputs)
    loss = loss + child_loss
  end
  local child_c, child_h = self:get_child_states(tree)
  self:allocate_module(tree, 'composer')
  tree.state = tree.composer:forward{inputs[tree.idx], child_c, child_h, inputs[idx_t]}

  if self.output_module ~= nil then
    self:allocate_module(tree, 'output_module')
    tree.output = tree.output_module:forward(tree.state[2])
    if self.train and tree.gold_label ~= nil then
      loss = loss + self.criterion:forward(tree.output, tree.gold_label)
    end
  end
  return tree.state, loss
end


function AttentionChildSumTreeLSTM:print(tree, depth, weight)
  for i =1,depth do printf("--") end
  printf("%s %f %s\n", tree.l, weight, tree.i)
  for i = 1, tree.num_children do
    self:print(tree.children[i], depth+1, tree.a[1][i])
  end

end

function AttentionChildSumTreeLSTM:getWeight(tree, idx_t, inputs, labels, i2s)
  for i = 1, tree.num_children do
    self:getWeight(tree.children[i], idx_t, inputs, labels, i2s)
  end

  local child_c, child_h = self:get_child_states(tree)
  self:allocate_module(tree, 'composer')
  tree.state = tree.composer:forward{inputs[tree.idx], child_c, child_h, inputs[idx_t]}

  tree.i = i2s[tree.idx]
  tree.l = labels[i2s[tree.idx]]
  tree.a = tree.composer:get(8).output:clone()
--  print(tree.l, tree.a)
  if self.output_module ~= nil then
    self:allocate_module(tree, 'output_module')
    tree.output = tree.output_module:forward(tree.state[2])
  end
end

function AttentionChildSumTreeLSTM:backward(tree, idx_t, inputs, grad)
  local grad_inputs = torch.Tensor(inputs:size())
  self:_backward(tree, idx_t, inputs, grad, grad_inputs)
  return grad_inputs
end

function AttentionChildSumTreeLSTM:_backward(tree, idx_t, inputs, grad, grad_inputs)
  local output_grad = self.mem_zeros
  if tree.output ~= nil and tree.gold_label ~= nil then
    output_grad = tree.output_module:backward(
      tree.state[2], self.criterion:backward(tree.output, tree.gold_label))
  end
  self:free_module(tree, 'output_module')
  tree.output = nil

  local child_c, child_h = self:get_child_states(tree)
  local composer_grad = tree.composer:backward(
    {inputs[tree.idx], child_c, child_h, inputs[idx_t]},
    {grad[1], grad[2] + output_grad})
  self:free_module(tree, 'composer')
  tree.state = nil

  grad_inputs[tree.idx] = composer_grad[1]
  local child_c_grads, child_h_grads = composer_grad[2], composer_grad[3]
  for i = 1, tree.num_children do
    self:_backward(tree.children[i], idx_t, inputs, {child_c_grads[i], child_h_grads[i]}, grad_inputs)
  end
end

function AttentionChildSumTreeLSTM:clean(tree)
  self:free_module(tree, 'composer')
  self:free_module(tree, 'output_module')
  tree.state = nil
  tree.output = nil
  for i = 1, tree.num_children do
    self:clean(tree.children[i])
  end
end

function AttentionChildSumTreeLSTM:parameters()
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

function AttentionChildSumTreeLSTM:get_child_states(tree)
  local child_c, child_h
  if tree.num_children == 0 then
    child_c = torch.zeros(1, self.mem_dim)
    child_h = torch.zeros(1, self.mem_dim)
  else
    child_c = torch.Tensor(tree.num_children, self.mem_dim)
    child_h = torch.Tensor(tree.num_children, self.mem_dim)
    for i = 1, tree.num_children do
       child_c[i], child_h[i] = unpack(tree.children[i].state)
    end
  end
  return child_c, child_h
end
