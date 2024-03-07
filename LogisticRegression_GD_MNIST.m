%% Loading MNIST data

load mnist-matlab-master/mnist.mat;



trian_data_raw  = training.images;
num_train_samples_all = training.count;
train_labels_all = training.labels;
feature_size = training.height * training.width;


test_data_raw = test.images;
num_test_samples_all = test.count;
test_labels_all = test.labels;



train_data_permuted = permute(trian_data_raw, [3, 1, 2]);
train_data_all = reshape(train_data_permuted, num_train_samples_all , feature_size);

test_data_permuted = permute(test_data_raw, [3, 1, 2]);
test_data_all = reshape(test_data_permuted, num_test_samples_all , feature_size);


class_labels = [3, 9]; % Pairs of classes to classify

train_data = [];
train_labels = [];
test_data = [];
test_labels = [];

% Changing labels to 0 and 1
% Extract images and labels for the current class pair
for i = 1:length(class_labels)
    indices = find(train_labels_all == class_labels(i));
    filtered_data = train_data_all(indices, :);
    labels = ones(size(filtered_data, 1), 1) * class_labels(i);
    
    train_data = [train_data; filtered_data];
    train_labels = [train_labels; labels];


    indices = find(test_labels_all == class_labels(i));
    filtered_data = test_data_all(indices, :);
    labels = ones(size(filtered_data, 1), 1) * class_labels(i);
    
    test_data = [test_data; filtered_data];
    test_labels = [test_labels; labels];
end

% Map class labels to 0 and 1
train_labels(train_labels == class_labels(1)) = 0;
train_labels(train_labels == class_labels(2)) = 1;

test_labels(test_labels == class_labels(1)) = 0;
test_labels(test_labels == class_labels(2)) = 1;

% Shuffle train data
rng(42); % Seed for reproducibility
shuffled_indices = randperm(size(train_labels, 1));
train_data = train_data(shuffled_indices, :);
train_labels = train_labels(shuffled_indices, :);

% Shuffle test data
rng(42); % Seed for reproducibility
shuffled_indices = randperm(size(test_labels, 1));
test_data = test_data(shuffled_indices, :);
test_labels = test_labels(shuffled_indices, :);


%%

num_iterations = 200;
num_train_samples = size(train_labels, 1);
num_test_samples = size(test_labels, 1);

% 
% Initialize the weights for logistic regression
theta = randn(size(train_data, 2) + 1, 1);

train_data = [train_data,ones(num_train_samples,1)];
test_data = [test_data,ones(num_test_samples ,1)];


% Initialize array to store the loss at each iteration
losses = zeros(num_iterations, 1);

%% Gradient Descent With Logistic Loss

% learning rate
alpha = 0.99;
for k = 1:num_iterations
    % run loss with current weights

    eps = 1e-5 ; 
    predictions = sigmoid(train_data*theta);
    predictions = max(min(predictions, 1-eps), eps);

    J = -train_labels.*log(sigmoid(predictions)) - (1-train_labels).*log(1-sigmoid(predictions));
    
    % update weights:
    gradient = 1/num_train_samples * train_data' * (sigmoid(train_data*theta)-train_labels);
    theta = theta - alpha * gradient;
end



% Calculate accuracy on training and test sets
train_predictions = 1./(1 + exp(-train_data * theta)) >= 0.5;
test_predictions = 1./(1 + exp(-test_data * theta)) >= 0.5;

train_accuracy = mean(train_predictions == train_labels);
test_accuracy = mean(test_predictions == test_labels);

fprintf('(Class %d vs Class %d):\n', class_labels(1), class_labels(2));
fprintf('Training Accuracy: %.2f%%\n', train_accuracy * 100);
fprintf('Test Accuracy: %.2f%%\n\n', test_accuracy * 100);

scaled_loss =  ((loss_train - min(loss_train)) / (max(loss_train) - min(loss_train)));

%%

% % Plot the loss over iterations
figure;
semilogy(1:num_iterations, scaled_loss, 'LineWidth', 2);
title(sprintf('Loss over Iterations for Class %d vs %d', class_labels( 1), class_labels(2)));
xlabel('Iteration');
ylabel('Loss');
grid on;

x = 1:num_iterations;
envelope  = 1./x;
hold on;

semilogy(envelope);

legend('Empirical', 'Theoritical');
    
%% Validation 

% Randomly select an image
randIndex = randi([1, num_test_samples], 1, 1);
selectedImage = test_data(randIndex, :);
true_label = test_labels(randIndex);

% Do prediction
sample_test_prediction = 1./(1 + exp(-selectedImage * theta)) >= 0.5;

selectedImage(end) = [];
% Reshape the selected image back to its original X-by-Y size
w = training.width;
h = training.height;
reshapedImage = reshape(selectedImage, w, h);



if true_label == 0
    true_label = class_labels(1);
else
    true_label = class_labels(2);
end

if sample_test_prediction == 0
    sample_test_prediction = class_labels(1);
else
    sample_test_prediction = class_labels(2);
end    


% Display the image
figure;
imagesc(reshapedImage); % Use imagesc for automatic scaling of the display
colormap('gray'); % Set the colormap to gray for a grayscale image
axis('image'); % Ensure the aspect ratio is correct
title(sprintf('Predicted label: %d, True label: %d', sample_test_prediction, true_label));

%%
function y = sigmoid(z)
    y = 1./(1+exp(-z));
end
