%特征提取，传入正负样本，提取特征后赋label标签并保存
% 清空环境变量
close all;
clear;
clc;
format compact

% % %Kmer
% system('python ./Pse-in-One-2.0/nac.pyc ./Temp_data/3133p_s.txt Protein Kmer -k 2 -f tab -labels 0 -out ./Temp_data/PCBmark_pos.txt');
% system('python ./Pse-in-One-2.0/nac.pyc ./Temp_data/3157n_s.txt Protein Kmer -k 2 -f tab -labels 0 -out ./Temp_data/PCBmark_neg.txt');
% system('python ./Pse-in-One-2.0/nac.pyc ./Temp_data/1360p_s.txt Protein Kmer -k 1 -f tab -labels 0 -out ./Temp_data/PCBmark_pos.txt');
% system('python ./Pse-in-One-2.0/nac.pyc ./Temp_data/1336n_s.txt Protein Kmer -k 1 -f tab -labels 0 -out ./Temp_data/PCBmark_neg.txt');
% %DR
% system('python2715 ./Pse-in-One-2.0/nac.py ./dataset/train/kcr_cvP_cleaned.txt Protein DR -max_dis 1 -f tab -labels 0 -out ./dataset/train/DR_kcr_cvP.txt');
% system('python2715 ./Pse-in-One-2.0/nac.py ./dataset/train/kcr_cvN_cleaned.txt Protein DR -max_dis 1 -f tab -labels 0 -out ./dataset/train/DR_kcr_cvN.txt');
system('python2715 ./Pse-in-One-2.0/nac.py ./dataset/test/Kcr_INDP_cleaned.txt Protein DR -max_dis 1 -f tab -labels 0 -out ./dataset/test/DR_Kcr_INDP.txt');
system('python2715 ./Pse-in-One-2.0/nac.py ./dataset/test/Kcr_INDN_cleaned.txt Protein DR -max_dis 1 -f tab -labels 0 -out ./dataset/test/DR_Kcr_INDN.txt');
%Distance Pair
% system('python ./Pse-in-One-2.0/nac.pyc ./Temp_data/3133p_s.txt Protein DR -max_dis 3 -f tab -labels 0 -out ./Temp_data/PCBmark_pos.txt');
% system('python ./Pse-in-One-2.0/nac.pyc ./Temp_data/3157n_s.txt Protein DR -max_dis 3 -f tab -labels 0 -out ./Temp_data/PCBmark_neg.txt');
% system('python ./Pse-in-One-2.0/nac.pyc ./Temp_data/Pos_samples.txt Protein DP -max_dis 4 -cp cp_19 -f tab -labels 0 -out ./Temp_data/PCBmark_pos.txt');
% system('python ./Pse-in-One-2.0/nac.pyc ./Temp_data/Neg_samples.txt Protein DP -max_dis 4 -cp cp_19 -f tab -labels 0 -out ./Temp_data/PCBmark_neg.txt');
%缺失
% %PC-PseAAC-General 缺失文件
% % system('python ./Pse-in-One-2.0/psee.pyc ./Temp_data/Pos_samples.txt Protein PC-PseAAC-General -i propChosen.txt -lamada 2 -w 0.1 -f tab -labels 0 -out ./Temp_data/PCBmark_pos.txt');
% % system('python ./Pse-in-One-2.0/psee.pyc ./Temp_data/Neg_samples.txt Protein PC-PseAAC-General -i propChosen.txt -lamada 2 -w 0.1 -f tab -labels 0 -out ./Temp_data/PCBmark_neg.txt');
% % system('python ./Pse-in-One-2.0/psee.pyc ./Temp_data/Pos_samples.txt Protein PC-PseAAC-General -i propChosen.txt -lamada 2 -w 0.1  -f tab -labels 0 -out ./Temp_data/PCBmark_pos.txt');
% % system('python ./Pse-in-One-2.0/psee.pyc ./Temp_data/Neg_samples.txt Protein PC-PseAAC-General -i propChosen.txt -lamada 2 -w 0.1  -f tab -labels 0 -out ./Temp_data/PCBmark_neg.txt');
%缺失
% %SC-PseAAC-General 缺失文件
% % system('python ./Pse-in-One-2.0/psee.pyc ./Temp_data/Pos_samples.txt Protein SC-PseAAC-General -i propChosen.txt -lamada 2 -w 0.1 -f tab -labels 0 -out ./Temp_data/PCBmark_pos.txt');
% % system('python ./Pse-in-One-2.0/psee.pyc ./Temp_data/Neg_samples.txt Protein SC-PseAAC-General -i propChosen.txt -lamada 2 -w 0.1 -f tab -labels 0 -out ./Temp_data/PCBmark_neg.txt');
% % system('python ./Pse-in-One-2.0/psee.pyc ./Temp_data/Pos_samples.txt Protein SC-PseAAC-General -i propChosen.txt -lamada 2 -w 0.1  -f tab -labels 0 -out ./Temp_data/PCBmark_pos.txt');
% % system('python ./Pse-in-One-2.0/psee.pyc ./Temp_data/Neg_samples.txt Protein SC-PseAAC-General -i propChosen.txt -lamada 2 -w 0.1  -f tab -labels 0 -out ./Temp_data/PCBmark_neg.txt');
% %PC-PseAAC
% system('python ./Pse-in-One/psee.pyc ./Temp_data/Pos_samples.txt Protein  -lamada 2 -w 0.1 -f tab -labels 0 -out ./Temp_data/PCBmark_pos.txt');
% system('python ./Pse-in-One-2.0/psee.pyc ./Temp_data/Neg_samples.txt Protein PC-PseAAC -lamada 2 -w 0.1 -f tab -labels 0 -out ./Temp_data/PCBmark_neg.txt');
% %SC-PseAAC
% system('python ./Pse-in-One-2.0/psee.pyc ./Temp_data/Pos_samples.txt Protein SC-PseAAC -lamada 2 -w 0.1 -f tab -labels 0 -out ./Temp_data/PCBmark_pos.txt');
% system('python ./Pse-in-One-2.0/psee.pyc ./Temp_data/Neg_samples.txt Protein SC-PseAAC -lamada 2 -w 0.1 -f tab -labels 0 -out ./Temp_data/PCBmark_neg.txt');
% %PDT？？？？
% system('python ./Pse-in-One-2.0/profile.py ./Temp_data/Pos_samples.txt Protein PDT -lamada 2 -f tab -labels 0 -out ./Temp_data/PCBmark_pos.txt');
% system('python ./Pse-in-One-2.0/profile.py ./Temp_data/Neg_samples.txt Protein PDT -lamada 2 -f tab -labels 0 -out ./Temp_data/PCBmark_neg.txt');
% %PDT-Profile?????
% system('python ./Pse-in-One-2.0/profile.pyc ./Temp_data/Pos_samples.txt Protein PDT-Profile -lamada 1 -n 1 -f tab -labels 0 -out ./Temp_data/PCBmark_pos.txt');
% system('python ./Pse-in-One-2.0/profile.pyc ./Temp_data/Neg_samples.txt Protein PDT-Profile -lamada 1 -n 1 -f tab -labels 0 -out ./Temp_data/PCBmark_neg.txt');
% %Top-n-gram?????
% system('python ./Pse-in-One-2.0/profile.py ./Temp_data/Pos_samples.txt Protein Top-n-gram -n 1 -f tab -labels 0 -out ./Temp_data/PCBmark_pos.txt');
% system('python ./Pse-in-One-2.0/profile.py ./Temp_data/Neg_samples.txt Protein Top-n-gram -n 1 -f tab -labels 0 -out ./Temp_data/PCBmark_neg.txt');
% %DT?????
% system('python ./Pse-in-One-2.0/profile.py ./Temp_data/Pos_samples.txt Protein DT -max_dis 1 -f tab -labels 0 -out ./Temp_data/PCBmark_pos.txt');
% system('python ./Pse-in-One-2.0/profile.py ./Temp_data/Neg_samples.txt Protein DT -max_dis 1 -f tab -labels 0 -out ./Temp_data/PCBmark_neg.txt');


filename = './dataset/test/Kcr_INDP_cleaned.txt'; % 正样本的txt文件路径
filename1 = './dataset/test/Kcr_INDN_cleaned.txt'; %负样本的txt文件路径
fileID = fopen(filename, 'r');
fileID1 = fopen(filename1, 'r');
data = textscan(fileID, '%s', 'Delimiter', '\n'); % 以换行符为分隔符读取每一行数据
data1 = textscan(fileID1, '%s', 'Delimiter', '\n'); % 以换行符为分隔符读取每一行数据
lines = data{1};
lines1 = data1{1};
fclose(fileID);
fclose(fileID1);

% 统计正样本每一行的数据数量
num_elements = zeros(length(lines), 1);
for i = 1:length(lines)
    line_cells = strsplit(lines{i}); % 将一行文本拆分为单元格数组
    num_elements(i) = length(line_cells); % 获取元素数量
end
% 统计负样本每一行的数据数量
num_elements1 = zeros(length(lines1), 1);
for i = 1:length(lines1)
    line_cells1 = strsplit(lines1{i}); % 将一行文本拆分为单元格数组
    num_elements1(i) = length(line_cells1); % 获取元素数量
end
% 显示每一行的数据数量
disp('正样本每一行的数据数量：');
%disp(num_elements);
disp('负样本每一行的数据数量：');
%disp(num_elements1);
% 读取txt文件
data = readtable('./dataset/test/DR_Kcr_INDP.txt', 'Delimiter', '\t');
data1 = readtable('./dataset/test/DR_Kcr_INDN.txt', 'Delimiter', '\t');
% 创建'label'列并赋值为"1"
data.label = repmat("1", height(data), 1);
% 创建'label'列并赋值为"0"
data1.label = repmat("0", height(data1), 1);
% 将txt数据保存为csv文件
writetable(data, './dataset/test/DR_Kcr_INDP.csv');
writetable(data1, './dataset/test/DR_Kcr_INDN.csv');

%合并特征提取后得到的正负样本数据集
% 读取第一个CSV文件
data1 = readtable('./dataset/test/DR_Kcr_INDP.csv');
% 读取第二个CSV文件
data2 = readtable('./dataset/test/DR_Kcr_INDN.csv');
% 合并两个数据表
mergedData = [data1; data2];
% 将合并后的数据保存为新的 CSV 文件
writetable(mergedData, './dataset/test/DR_Kcr_IND.csv');
%程序结束
disp('over');




