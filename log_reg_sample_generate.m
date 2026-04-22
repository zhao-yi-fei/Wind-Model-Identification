clc; clear; close all;
rng(42);

%% ===================== Configuration =====================
x = linspace(-500, 500, 100);
y = linspace(-500, 500, 100);
h = linspace(0, 500, 100);
[xw, yw, hw] = meshgrid(x, y, h);

[ny, nx, nz] = size(xw);

num_samples = 1000; 

noise_amp = 0.1;

% Allocate memory, final shape: [N, y, x, h, 3]
final_data = zeros(num_samples, ny, nx, nz, 3, 'single');
% Parameter labels
param_labels = zeros(num_samples, 4, 'single');

%% ===================== Generate Latin hypercube parameters =====================
% Logarithmic wind parameters: [H_R, V_R, h0, psi]
log_lhs = lhsdesign(num_samples, 4, 'criterion', 'maximin', 'iterations', 1000);
log_params = zeros(num_samples, 4);
log_params(:,3) = max(log_lhs(:,3) * 2, 0.1);  % h0(void too small values): 0.1~2
log_params(:,1) = log_params(:,3) + 0.1 + (500 - log_params(:,3) - 0.1) .* log_lhs(:,1); % H_R: 0.1~500
log_params(:,2) = log_lhs(:,2) * 25 + 5;       % V_R: 5~30
log_params(:,4) = log_lhs(:,4) * 2 * pi;       % psi: 0~2pi

%% ===================== Data generation =====================
idx = 0;

for sample = 1:num_samples
    idx = idx + 1;

    H_R = log_params(sample,1);
    V_R = log_params(sample,2);
    h_0 = log_params(sample,3);
    psi = log_params(sample,4);

    scale = log(hw / h_0) / log(H_R / h_0);
    U = V_R * scale * cos(psi);
    V = V_R * scale * sin(psi);
    W = zeros(size(U));
    
    % add noise
    % U = single(U + (rand(size(U))*2 - 1) * noise_amp);
    % V = single(V + (rand(size(V))*2 - 1) * noise_amp);
    % W = single(W + (rand(size(W))*2 - 1) * noise_amp);

    final_data(idx, :, :, :, 1) = reshape(U, [1, ny, nx, nz]);
    final_data(idx, :, :, :, 2) = reshape(V, [1, ny, nx, nz]);
    final_data(idx, :, :, :, 3) = reshape(W, [1, ny, nx, nz]);
    param_labels(idx,1:4) = [H_R, V_R, h_0, psi];
end
    
%% ===================== Save data =====================
addpath('npy-matlab-master\npy-matlab');

writeNPY(final_data,    'E:\log-wind\log_wind_samples1000.npy');
writeNPY(param_labels, 'E:\log-wind\log_wind_param1000.npy');

%% ===================== Verify output =====================
disp(['Data shape: ', mat2str(size(final_data))]);      % [1000,100,100,100,3]
disp(['Label shape: ', mat2str(size(param_labels))]);   % [1000,4]
