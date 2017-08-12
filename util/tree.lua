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

  A basic tree structure.

--]]

local Tree = torch.class('util.Tree')

function Tree:__init()
  self.parent = nil
  self.num_children = 0
  self.children = {}
end

function Tree:add_child(c)
  c.parent = self
  self.num_children = self.num_children + 1
  self.children[self.num_children] = c
end

function Tree:size()
  if self._size ~= nil then return self._size end
  local size = 1
  for i = 1, self.num_children do
    size = size + self.children[i]:size()
  end
  self._size = size
  return size
end

function Tree:depth()
  local depth = 0
  if self.num_children > 0 then
    for i = 1, self.num_children do
      local child_depth = self.children[i]:depth()
      if child_depth > depth then
        depth = child_depth
      end
    end
    depth = depth + 1
  end
  return depth
end


local function depth_first_preorder(tree, nodes)
  table.insert(nodes, tree.idx)
  for i = 1, tree.num_children do
      depth_first_preorder(tree.children[i], nodes)
  end
end

function Tree:depth_first_preorder()
  local nodes = {}
  depth_first_preorder(self, nodes)
  return nodes
end

