
function eval.accuracy(pred, gold)
    local count = 0
    local size = pred:size()[1]
    for i = 1, size do
        if pred[i] == gold[i] then
            count = count + 1
        end
    end
    return count / size
end

