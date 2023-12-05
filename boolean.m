% function main()
    % Loop over dimensions 2, 3, 4, 5
    for n = 2:5
        unique_functions = containers.Map;
        num_linearly_separable = 0;
        
        for iteration = 1:10000
            % Generate a random Boolean function
            func = randi([0, 1], 1, 2^n);
            func_str = mat2str(func);
            
            % Check if function is unique
            if isKey(unique_functions, func_str)
                continue;
            end
            
            % Add function to unique list
            unique_functions(func_str) = 1;
            
            % Generate data
            X = dec2bin(0:(2^n - 1)) - '0';
            Y = func;
            
            % Initialize weights and thresholds
            w = normrnd(0, sqrt(1/n), [1, n]);
            theta = 0;
            
            % Initialize learning rate
            eta = 0.05;
            
            % Train for 20 epochs
            is_correct = false;
            for epoch = 1:20
                epoch_error = 0;
                for i = 1:size(X, 1)
                    % Compute output
                    b = sum(w .* X(i, :)) - theta;
                    O = sign(b);
                    
                    % Target output
                    t = Y(i) * 2 - 1; % Convert 0 to -1 and 1 to 1
                    
                    % Update weights and threshold
                    delta = eta * (t - O);
                    w = w + delta * X(i, :);
                    theta = theta - delta;
                    
                    epoch_error = epoch_error + abs(t - O);
                end
                
                % If correctly classified all points, function is linearly separable
                if epoch_error == 0
                    is_correct = true;
                    break;
                end
            end
            
            if is_correct
                num_linearly_separable = num_linearly_separable + 1;
            end
        end
        
        fprintf('Number of linearly separable functions for n = %d: %d\n', n, num_linearly_separable);
    end
% end


