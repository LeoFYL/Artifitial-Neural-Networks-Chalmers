% uploading data
training_set = table2array(readtable('training_set.csv'));
validation_set = table2array(readtable('validation_set.csv'));

% standerlizaiton of traning data: mean = 0; std = 1;
mean_traning_set = mean(training_set);
std_traning_set = std(training_set);

Training = zeros(size(training_set));
Training(:,1) = (training_set(:,1)-mean_traning_set(1))/std_traning_set(1);
Training(:,2) = (training_set(:,2)-mean_traning_set(2))/std_traning_set(2);
Training(:,3) = training_set(:,3);


% standerlizaiton of validation data: mean = 0; std = 1;

mean_validation_set = mean(validation_set);
std_validation_set = std(validation_set);

Validation = zeros(size(validation_set));
Validation(:,1) = (validation_set(:,1)-mean_validation_set(1))/std_validation_set(1);
Validation(:,2) = (validation_set(:,2)-mean_validation_set(2))/std_validation_set(2);
Validation(:,3) = validation_set(:,3);



% Initialazion 

M1=[30 35 40];   % numbers of hidden neurons 
v_j = zeros(M1(1),1);
o_i = 0;
theta_1 = 0;       
theta_2 = 0;

eta = 0.04; % learning rate

w_jk = normrnd(0,1/sqrt(M1(1)),[M1(1),2]); % k = # of input = 2; j = # of hidden neurons;
w_ij = normrnd(0,1,[1,M1(1)]);  % i = # of output = 1;
delta_w_ij = zeros(size(w_ij));
delta_w_jk = zeros(size(w_jk));

% start training
n = 10000; % # of randomly draw pattern from the training set
epoch = 200;
for j =1:epoch
    for i = 1:n
        % computing propagate forward
        random_input = randi([1 size(Training(:,1),1)], 1); 
        x_k = (Training(random_input, 1:2))';
        t_i = Training(random_input, 3);
        v_j = tanh(w_jk * x_k - theta_1');
        o_i = tanh(w_ij * v_j - theta_2);
    
        % computing propagate backward
        delta_w_ij = (t_i-o_i) * (1-o_i^2);
        w_ij = w_ij + eta * delta_w_ij * (1-o_i^2) * (v_j)' ;

        delta_w_jk = delta_w_ij * w_ij .* (1-v_j.^2)';
        w_jk = w_jk + eta * (delta_w_jk)' * (x_k)';
    
        theta_2 = theta_2 - eta * delta_w_ij;
        theta_1 = theta_1 - eta * delta_w_jk;
    end

    % start the validating of the trained w_ij and w_jk.
    Pval = size(Validation(:,1),1);   % # of validation set
    count = 0;
    for m = 1: Pval
        v_x_k = (Validation(m, 1:2))';
        v_v_j = tanh(w_jk * v_x_k - theta_1');
        v_o_i = tanh(w_ij * v_v_j - theta_2);
        v_t_i = Validation(m,3);
        count = abs(sign(v_o_i) - v_t_i) + count;
    end
    classification_error = (1/(2*Pval) )* count;
    disp(['Epoch:  ' num2str(j)]);
    disp(['The error rate is:  ' num2str(classification_error)]);

    % keep record of w_ij, w_jk, theta_1, theta_2. 
    % if the error rate is lower than 12%, end the loop.
    if classification_error < 0.12
        csvwrite('w1.csv',w_jk);
        csvwrite('w2.csv',w_ij);
        csvwrite('t1.csv',theta_1);
        csvwrite('t2.csv',theta_2);
        break;
    end


end



     




