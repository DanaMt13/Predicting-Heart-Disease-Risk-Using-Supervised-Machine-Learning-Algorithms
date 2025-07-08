function [precision, recall, F1, accuracy] = computeMetrics(yTrue, yPred)
    % Calculăm matricea de confuzie
    C = confusionmat(yTrue, yPred);
    % Presupunând ca eticheta 0 este negativă și 1 este pozitivă:
    TP = C(2,2);
    TN = C(1,1);
    FP = C(1,2);
    FN = C(2,1);
    
    % Calculează metricele
    precision = TP / (TP + FP);
    recall    = TP / (TP + FN);
    F1        = 2 * (precision * recall) / (precision + recall);
    accuracy  = (TP + TN) / sum(C(:));
end