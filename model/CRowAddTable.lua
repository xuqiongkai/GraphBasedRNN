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

  Add a vector to every row of a matrix.

  Input: { [n x m], [m] }

  Output: [n x m]

--]]

local CRowAddTable, parent = torch.class('classifier.CRowAddTable', 'nn.Module')

function CRowAddTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CRowAddTable:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])
   for i = 1, self.output:size(1) do
      self.output[i]:add(input[2])
   end
   return self.output
end

function CRowAddTable:updateGradInput(input, gradOutput)
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[2] = self.gradInput[2] or input[2].new()
   self.gradInput[1]:resizeAs(input[1])
   self.gradInput[2]:resizeAs(input[2]):zero()

   self.gradInput[1]:copy(gradOutput)
   for i = 1, gradOutput:size(1) do
      self.gradInput[2]:add(gradOutput[i])
   end

   return self.gradInput
end
