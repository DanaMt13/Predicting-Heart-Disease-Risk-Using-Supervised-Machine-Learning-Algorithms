%%
clc
clear all
close all

%% 📌 1. Încărcare set de date
pTest = 0.2; % Proporția de date pentru testare
fileNameRawData = 'heart_failure.csv';
Z = readtable(fileNameRawData); % Citirea datelor din CSV

% 📊 Numărăm pacienții sănătoși și cei cu boală cardiacă
numPacientiSanatosi = sum(Z.HeartDisease == 0); % 410
numPacientiBolnavi = sum(Z.HeartDisease == 1);   % 508

% 📈 Creăm o diagramă pentru distribuția bolii cardiace
figure;
bar([numPacientiSanatosi, numPacientiBolnavi]);
set(gca, 'XTickLabel', {'Sănătoși (0)', 'Bolnavi (1)'});
title('Distribuția HeartDisease');
ylabel('Număr de pacienți');

%% 📌 2. Preprocesare date
% Extragem caracteristici numerice relevante
X_numeric = table2array(Z(:, {'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak'}));

% Conversie variabile categorice în valori numerice
X_sex = double(categorical(Z.Sex));
X_chestPain = double(categorical(Z.ChestPainType));
X_restingECG = double(categorical(Z.RestingECG));
X_exerciseAngina = double(categorical(Z.ExerciseAngina));
X_stSlope = double(categorical(Z.ST_Slope));

% Ordinea trasăturilor în X:
% Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak, Sex, ChestPainType, RestingECG, ExerciseAngina, ST_S_l_o_p_e.
X = [X_numeric, X_sex, X_chestPain, X_restingECG, X_exerciseAngina, X_stSlope];

% Extragem variabila țintă HeartDisease
Y = table2array(Z(:, 'HeartDisease'));

%% 📌 3. Gestionare valori lipsă
X(isnan(X)) = 0; % Înlocuim valorile lipsă cu 0

%% 📌 4. Calcularea și vizualizarea matricilor de corelație

% --- Pentru setul complet (predictori + target) ---
Y_scaled = double(Z.HeartDisease); % Asigurăm conversia targetului în numeric
X_with_target = [X_numeric, X_sex, X_chestPain, X_restingECG, X_exerciseAngina, X_stSlope, Y_scaled]; 

correlationMatrix_incl = corrcoef(X_with_target);
numFeatures_incl = size(X_with_target,2);
disp(['Dimensiune matrice corelație (inclusiv targetul): ', num2str(numFeatures_incl), 'x', num2str(numFeatures_incl)]);

featureNames_incl = {'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', ...
                     'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_S_l_o_p_e', 'HeartDisease'};
featureNames_incl = featureNames_incl(1:numFeatures_incl);

figure;
imagesc(correlationMatrix_incl);
colorbar;
title('Correlation Matrix - Heart Failure Prediction (Including HeartDisease)');
xticks(1:numFeatures_incl);
yticks(1:numFeatures_incl);
xticklabels(featureNames_incl);
yticklabels(featureNames_incl);
xtickangle(45);
ytickangle(45);

% --- Pentru doar predictorii (excludem targetul) ---
correlationMatrix_excl = corrcoef(X); 
numFeatures_excl = size(X,2);
disp(['Dimensiune matrice corelație (doar predictori): ', num2str(numFeatures_excl), 'x', num2str(numFeatures_excl)]);

featureNames_excl = {'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', ...
                     'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_S_l_o_p_e'};
featureNames_excl = featureNames_excl(1:numFeatures_excl);

figure;
imagesc(correlationMatrix_excl);
colorbar;
title('Correlation Matrix - Predictors Only');
xticks(1:numFeatures_excl);
yticks(1:numFeatures_excl);
xticklabels(featureNames_excl);
yticklabels(featureNames_excl);
xtickangle(45);
ytickangle(45);

% Antrenăm un model de arbore de decizie cu funcția fitctree pe setul de antrenare
% (Acest model ne ajută să evidențiem atributele care influențează decizia)
mdl_tree = fitctree(X, Y);

% Calculăm importanța predictorilor din modelul de arbore
importance_tree = predictorImportance(mdl_tree);

% Sortăm importanțele și afișăm un tabel
[sortedImp, sortedIdx] = sort(importance_tree, 'descend');

disp('Importanța predictorilor conform modelului de arbore de decizie:');
T_imp = table(sortedIdx, sortedImp, featureNames_excl(sortedIdx(:)), ...
              'VariableNames', {'Index', 'Importance', 'Feature'});

disp(T_imp);

%% 📌 5. Eliminarea variabilelor puternic corelate
threshold = 0.9;
highCorr_excl = abs(correlationMatrix_excl) > threshold;
[i_excl, j_excl] = find(triu(highCorr_excl, 1));  % Partea superioară
disp('Variabile cu corelație puternică între predictori (Fără target):');
disp([i_excl, j_excl]);
disp('(-, dacă nu se trece peste pragul impus)');

%% 📌 6. Distribuția variabilei țintă (HeartDisease)
tabulate(Y); % Afișează distribuția: 44.66% pacienți fără boală (0) și 55.34% cu boală (1)

%% 📌 7. Vizualizarea intervalului datelor prin histograme
featureNames = {'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', ...
                'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_S_l_o_p_e'};

mPlot = 3; nPlot = 4;
figure;
for k = 1:size(X,2)
    subplot(mPlot, nPlot, k)
    histogram(X(:,k))
    title(['Feature: ', featureNames{k}])
end

%% 📌 8. Normalizarea datelor
[X, Y] = normalizeData(X, Y, 0, 1); % Normalizează datele între [0,1]

figure;
for k = 1:size(X,2)
    subplot(mPlot, nPlot, k)
    histogram(X(:,k))
    title(['Feature: ', featureNames{k}])
end
sgtitle('Normalizarea datelor');

%% 📌 9. Împărțirea datasetului în antrenare și testare
pVal = 0; % Nu folosim set de validare
[XTrain, YTrain, XTest, YTest, XVal, YVal] = buildDatasets(X, Y, pTest, pVal);

%% 🚀 10. Antrenare Model K-NN (5 vecini)
tic;  % Pornește cronometrarea pentru antrenarea K-NN (5 vecini)
k5 = 5; 
mdl_knn_5 = fitcknn(XTrain, YTrain, 'NumNeighbors', k5);
trainingTime_KNN5 = toc;  % Timpul de antrenare pentru K-NN (5 vecini)
disp(['Training time for K-NN (5 Neighbors): ', num2str(trainingTime_KNN5), ' sec']);

%% 🚀 11. Antrenare Model K-NN (10 vecini)
tic;  % Pornește cronometrarea pentru antrenarea K-NN (10 vecini)
k10 = 10; 
mdl_knn_10 = fitcknn(XTrain, YTrain, 'NumNeighbors', k10);
trainingTime_KNN10 = toc;  % Timpul de antrenare pentru K-NN (10 vecini)
disp(['Training time for K-NN (10 Neighbors): ', num2str(trainingTime_KNN10), ' sec']);

%% 🚀 12. Antrenare Model Random Forest cu 50 arbori
tic;  % Pornește cronometrarea pentru Random Forest cu 50 arbori
numTrees_50 = 50; 
mdl_rf_50 = TreeBagger(numTrees_50, XTrain, YTrain, 'Method', 'classification');
trainingTime_RF50 = toc;  % Timpul de antrenare pentru Random Forest (50 arbori)
disp(['Training time for Random Forest (50 Trees): ', num2str(trainingTime_RF50), ' sec']);

%% 🚀 13. Antrenare Model Random Forest cu 200 arbori
tic;  % Pornește cronometrarea pentru Random Forest cu 200 arbori
numTrees_200 = 200; 
mdl_rf_200 = TreeBagger(numTrees_200, XTrain, YTrain, 'Method', 'classification');
trainingTime_RF200 = toc;  % Timpul de antrenare pentru Random Forest (200 arbori)
disp(['Training time for Random Forest (200 Trees): ', num2str(trainingTime_RF200), ' sec']);

%% 📊 14. Evaluare modele

% Predicții K-NN (5 vecini)
tic;  % Măsurăm timpul de predicție
YMTrain_KNN_5 = predict(mdl_knn_5, XTrain);
YMTest_KNN_5 = predict(mdl_knn_5, XTest);
predictionTime_KNN5 = toc;
disp(['Prediction time for K-NN (5 Neighbors): ', num2str(predictionTime_KNN5), ' sec']);

% Predicții K-NN (10 vecini)
tic;
YMTrain_KNN_10 = predict(mdl_knn_10, XTrain);
YMTest_KNN_10 = predict(mdl_knn_10, XTest);
predictionTime_KNN10 = toc;
disp(['Prediction time for K-NN (10 Neighbors): ', num2str(predictionTime_KNN10), ' sec']);

% Predicții Random Forest (50 arbori)
tic;
YMTrain_RF_50 = str2double(predict(mdl_rf_50, XTrain)); 
YMTest_RF_50 = str2double(predict(mdl_rf_50, XTest));
predictionTime_RF50 = toc;
disp(['Prediction time for Random Forest (50 Trees): ', num2str(predictionTime_RF50), ' sec']);

% Predicții Random Forest (200 arbori)
tic;
YMTrain_RF_200 = str2double(predict(mdl_rf_200, XTrain)); 
YMTest_RF_200 = str2double(predict(mdl_rf_200, XTest));
predictionTime_RF200 = toc;
disp(['Prediction time for Random Forest (200 Trees): ', num2str(predictionTime_RF200), ' sec']);

%% 📌 15. Vizualizare rezultate
% Matrice de confuzie pentru fiecare model
figure, confusionchart(YTest, YMTest_KNN_5), title('K-NN (5 Neighbors) - Confusion Matrix');
figure, confusionchart(YTest, YMTest_KNN_10), title('K-NN (10 Neighbors) - Confusion Matrix');
figure, confusionchart(YTest, YMTest_RF_50), title('Random Forest (50 Trees) - Confusion Matrix');
figure, confusionchart(YTest, YMTest_RF_200), title('Random Forest (200 Trees) - Confusion Matrix');%partea stanga sus- cazurile in care clasa reala a fost 0 si modelul a
%prezis corect tot 0 (True Negatives)

%partea dreapta jos- cazurile în care clasa reală a fost 1 și modelul a
% prezis corect tot 1 (True Positives).

%partea dreapta sus-cazurile în care clasa reală a fost 0, dar modelul a prezis
%greșit 1 (False Positives).

%partea stanga jos-cazurile în care clasa reală a fost 1, dar modelul a prezis
% greșit 0 (False Negatives).


% Calculul metricilor pentru K-NN (5 vecini)
C_KNN_5 = confusionmat(YTest, YMTest_KNN_5);
TP = C_KNN_5(2,2); TN = C_KNN_5(1,1); FP = C_KNN_5(1,2); FN = C_KNN_5(2,1);
precision_KNN_5 = TP / (TP + FP);
recall_KNN_5    = TP / (TP + FN);
F1_KNN_5        = 2 * (precision_KNN_5 * recall_KNN_5) / (precision_KNN_5 + recall_KNN_5);
accuracy_KNN_5  = (TP + TN) / sum(C_KNN_5(:));

disp('--- K-NN (5 Neighbors) on Test Set ---');
disp(['Precision: ', num2str(precision_KNN_5)]);
disp(['Recall:    ', num2str(recall_KNN_5)]);
disp(['F1 Score:  ', num2str(F1_KNN_5)]);
disp(['Accuracy:  ', num2str(accuracy_KNN_5)]);

% Calculul metricilor pentru K-NN (10 vecini)
C_KNN_10 = confusionmat(YTest, YMTest_KNN_10);
TP = C_KNN_10(2,2); TN = C_KNN_10(1,1); FP = C_KNN_10(1,2); FN = C_KNN_10(2,1);
precision_KNN_10 = TP / (TP + FP);
recall_KNN_10    = TP / (TP + FN);
F1_KNN_10        = 2 * (precision_KNN_10 * recall_KNN_10) / (precision_KNN_10 + recall_KNN_10);
accuracy_KNN_10  = (TP + TN) / sum(C_KNN_10(:));

disp('--- K-NN (10 Neighbors) on Test Set ---');
disp(['Precision: ', num2str(precision_KNN_10)]);
disp(['Recall:    ', num2str(recall_KNN_10)]);
disp(['F1 Score:  ', num2str(F1_KNN_10)]);
disp(['Accuracy:  ', num2str(accuracy_KNN_10)]);

% Calculul metricilor pentru Random Forest (50 arbori)
C_RF_50 = confusionmat(YTest, YMTest_RF_50);
TP = C_RF_50(2,2); TN = C_RF_50(1,1); FP = C_RF_50(1,2); FN = C_RF_50(2,1);
precision_RF_50 = TP / (TP + FP);
recall_RF_50    = TP / (TP + FN);
F1_RF_50        = 2 * (precision_RF_50 * recall_RF_50) / (precision_RF_50 + recall_RF_50);
accuracy_RF_50  = (TP + TN) / sum(C_RF_50(:));

disp('--- Random Forest (50 Trees) on Test Set ---');
disp(['Precision: ', num2str(precision_RF_50)]);
disp(['Recall:    ', num2str(recall_RF_50)]);
disp(['F1 Score:  ', num2str(F1_RF_50)]);
disp(['Accuracy:  ', num2str(accuracy_RF_50)]);

% Calculul metricilor pentru Random Forest (200 arbori)
C_RF_200 = confusionmat(YTest, YMTest_RF_200);
TP = C_RF_200(2,2); TN = C_RF_200(1,1); FP = C_RF_200(1,2); FN = C_RF_200(2,1);
precision_RF_200 = TP / (TP + FP);
recall_RF_200    = TP / (TP + FN);
F1_RF_200        = 2 * (precision_RF_200 * recall_RF_200) / (precision_RF_200 + recall_RF_200);
accuracy_RF_200  = (TP + TN) / sum(C_RF_200(:));

disp('--- Random Forest (200 Trees) on Test Set ---');
disp(['Precision: ', num2str(precision_RF_200)]);
disp(['Recall:    ', num2str(recall_RF_200)]);
disp(['F1 Score:  ', num2str(F1_RF_200)]);
disp(['Accuracy:  ', num2str(accuracy_RF_200)]);

