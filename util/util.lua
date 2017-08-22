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

function util.extract_content(content_path, feature_path, label_path, meta_path)
    --printf("%s, %s, %s.\n", feature_path, label_path, meta_path)
    local content_file = io.open(content_path, 'r')
    local line, tokens
    local dataset = {}
    local sample_num, feature_num = 0, 0

    -- write label and calculate sample number
    local label_file = io.open(label_path, 'w')
    local meta_file = io.open(meta_path, 'w')
    local label_set = {}
    while true do
        line = content_file:read()
        if line == nil  then break end
        tokens = stringx.split(line, "\t")
        sample_num = sample_num + 1
        feature_num = #tokens - 2
        label_file:write(tokens[1].."\t"..tokens[#tokens].."\n")
        label_set[tokens[#tokens]] = 1
    end
    content_file:close()
    label_file:close()

    --write to meta data
    for k, _ in pairs(label_set) do
        meta_file:write(k.."\n")
    end
    meta_file:close()


    -- extract features
    local features = torch.Tensor(sample_num, feature_num)
    sample_num = 0
    content_file = io.open(content_path, 'r')
    while true do
        line = content_file:read()
        if line == nil  then break end
        tokens = stringx.split(line, "\t")
        sample_num = sample_num + 1
        for i = 2, #tokens - 1 do
            features[sample_num][i-1] = tonumber(tokens[i])
        end
    end
    content_file:close()
    torch.save(feature_path, features)
end

function util.read_features(feature_path)
    return torch.load(feature_path)
end
function util.read_mat_features(feature_path)
    local sample_num, feature_num = 0, 0
    local feature_file = io.open(feature_path, 'r')
    local line, tokens
    while true do
        line = feature_file:read()
        if line == nil  then break end
        tokens = stringx.split(line, ",")
        feature_num = #tokens
        sample_num = sample_num + 1
    end
    feature_file:close()

    local features = torch.DoubleTensor(sample_num, feature_num)
    sample_num = 0
    feature_file = io.open(feature_path, 'r')
    while true do
        line = feature_file:read()
        if line == nil  then break end
        tokens = stringx.split(line, ",")
        sample_num = sample_num + 1
        for i = 1, #tokens do
            features[sample_num][i] = tonumber(tokens[i])
        end
    end
    return features
end

function util.read_features_text(feature_path, to_feature_path)
    local sample_num, feature_num = 0, 0
    local feature_file = io.open(feature_path, 'r')
    local line, tokens
    while true do
        line = feature_file:read()
        if line == nil  then break end
        tokens = stringx.split(line, "\t")
        feature_num = #tokens
        sample_num = sample_num + 1
    end
    feature_file:close()

    local features = torch.DoubleTensor(sample_num, feature_num)
    sample_num = 0
    feature_file = io.open(feature_path, 'r')
    while true do
        line = feature_file:read()
        if line == nil  then break end
        tokens = stringx.split(line, "\t")
        sample_num = sample_num + 1
        for i = 1, #tokens do
            features[sample_num][i] = tonumber(tokens[i])
        end
    end
    torch.save(to_feature_path, features)
end

function util.read_labels(label_path)
    local label_file = io.open(label_path, 'r')
    local i2s, s2i = {}, {}
    local labels = {}
    local line, tokens
    local count = 0
    while true do
        line = label_file:read()
        if line == nil  then break end
        count = count + 1

        tokens = stringx.split(line, "\t")
        i2s[count] = tokens[1]
        s2i[tokens[1]] = count
        labels[tokens[1]] = tokens[2]
    end
    label_file:close()
    return i2s, s2i, labels
end

function util.read_scores(label_path)
    local label_file = io.open(label_path, 'r')
    local i2s, s2i = {}, {}
    local labels = {}
    local line, tokens
    local count = 0
    while true do
        line = label_file:read()
        if line == nil  then break end
        count = count + 1

        tokens = stringx.split(line, "\t")
        i2s[count] = tokens[1]
        s2i[tokens[1]] = count
        labels[tokens[1]] = torch.Tensor(1):fill(tonumber(tokens[2]))
    end
    label_file:close()
    return i2s, s2i, labels
end

function util.read_meta(meta_path)
    local meta_file = io.open(meta_path, 'r')
    local map = {}
    local count = 0
    local line
    while true do
        line = meta_file:read()
        if line == nil  then break end
        count = count + 1
        map[line] = count
    end
    meta_file:close()
    return map, count
end

function util.read_cites(cites_path)
    local cites_file = io.open(cites_path, 'r')
    local line, tokens
    local dataset = {}
    local data
    while true do
        line = cites_file:read()
        if line == nil  then break end
        tokens = stringx.split(line, "\t")
        data = dataset[tokens[1]]
        if data == nil then data = {} end
        data[#data + 1] = tokens[2]
        dataset[tokens[1]] = data
    end
    return dataset

end


function util.generate_indices(sample_num, alpha)
    local indices = torch.randperm(sample_num)
    --local indices = torch.range(1, sample_num)
    local train_num = torch.floor(sample_num * alpha)

    local train_indices = torch.range(1, train_num)
    local test_indices = torch.range(train_num + 1, sample_num)
    local train_indices_set = {}
    local test_indices_set = {}
    for i = 1,train_indices:size()[1] do
        train_indices[i] = indices[train_indices[i]]
        train_indices_set[train_indices[i]] = true
    end
    for i = 1,test_indices:size()[1] do
        test_indices[i] = indices[test_indices[i]]
        test_indices_set[test_indices[i]] = true
    end
    return train_indices, test_indices, train_indices_set, test_indices_set
end

function util.best_label(output, label_num)
    local prediction = 1
    local best_score = -1000
    for i = 1, label_num do
        if output[i] > best_score then
            best_score = output[i]
            prediction = i
        end
    end
    return prediction
end

function util.split_string(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end