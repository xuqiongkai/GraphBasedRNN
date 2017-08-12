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

  Tree-LSTM base class

--]]

local TreeLSTM, parent = torch.class('classifier.TreeLSTM', 'nn.Module')

function TreeLSTM:__init(config)
  parent.__init(self)
  self.in_dim = config.in_dim
  if self.in_dim == nil then error('input dimension must be specified') end
  self.mem_dim = config.mem_dim or 150
  self.mem_zeros = torch.zeros(self.mem_dim)
  self.train = false
end

function TreeLSTM:forward(tree, inputs)
end

function TreeLSTM:backward(tree, inputs, grad)
end

function TreeLSTM:training()
  self.train = true
end

function TreeLSTM:evaluate()
  self.train = false
end

function TreeLSTM:allocate_module(tree, module)
  local modules = module .. 's'
  local num_free = #self[modules]
  if num_free == 0 then
    tree[module] = self['new_' .. module](self)
  else
    tree[module] = self[modules][num_free]
    self[modules][num_free] = nil
  end

  -- necessary for dropout to behave properly
  if self.train then tree[module]:training() else tree[module]:evaluate() end
end

function TreeLSTM:free_module(tree, module)
  if tree[module] == nil then return end
  table.insert(self[module .. 's'], tree[module])
  tree[module] = nil
end
