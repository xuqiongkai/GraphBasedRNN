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


require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

classifier = {}
cora = {}
yelp = {}
util = {}
eval = {}


include('util/util.lua')
include('util/evaluation.lua')
include('util/tree.lua')

include('model/CRowAddTable.lua')
include('model/TreeLSTM.lua')
include('model/ChildSumTreeLSTM.lua')
include('model/ChildSumTreeRNN.lua')
include('model/AttentionChildSumTreeLSTM.lua')

include('main/LogisticModel.lua')
include('main/IterLogisticModel.lua')
include('main/LabelPropogationModel.lua')
include('main/GraphRNN.lua')

printf = utils.printf



-- share module parameters
function share_params(cell, src)
  if torch.type(cell) == 'nn.gModule' then
    for i = 1, #cell.forwardnodes do
      local node = cell.forwardnodes[i]
      if node.data.module then
        node.data.module:share(src.forwardnodes[i].data.module,
          'weight', 'bias', 'gradWeight', 'gradBias')
      end
    end
  elseif torch.isTypeOf(cell, 'nn.Module') then
    cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
  else
    error('parameters cannot be shared for this input')
  end
end

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end








