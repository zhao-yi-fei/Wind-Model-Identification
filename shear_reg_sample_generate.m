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
param_labels = zeros(num_samples, 5, 'single');

%% ===================== Generate Latin hypercube parameters =====================
% Shear layer parameters: [V_0, V_R, Delta_h, h0, psi]
fqb_lhs = lhsdesign(num_samples, 5, 'criterion', 'maximin', 'iterations', 1000);
fqb_params = zeros(num_samples, 5);
fqb_params(:,1) = fqb_lhs(:,1) * 5 + 5;      % V_0: 5~10
fqb_params(:,2) = fqb_lhs(:,2) * 20 + 5;     % V_R: 5~25
fqb_params(:,3) = fqb_lhs(:,3) * 350 + 50;   % Delta_h: 50~400
fqb_params(:,4) = fqb_lhs(:,4) * 90 + 10;    % h0: 10~100
fqb_params(:,5) = fqb_lhs(:,5) * 2 * pi;     % psi: 0~2pi

%% ===================== Data generation =====================
idx = 0;

for sample = 1:num_samples
    idx = idx + 1;

    V_0     = fqb_params(sample,1);
    V_R     = fqb_params(sample,2);
    Delta_h = fqb_params(sample,3);
    h_0     = fqb_params(sample,4);
    psi     = fqb_params(sample,5);

    scale = V_0 + V_R ./ (1 + exp((14 / Delta_h) * (Delta_h / 2 - (hw - h_0))));
    U = scale * cos(psi);
    V = scale * sin(psi);
    W = zeros(size(U));
    
    % add noise
    % U = single(U + (rand(size(U))*2 - 1) * noise_amp);
    % V = single(V + (rand(size(V))*2 - 1) * noise_amp);
    % W = single(W + (rand(size(W))*2 - 1) * noise_amp);

    final_data(idx, :, :, :, 1) = reshape(U, [1, ny, nx, nz]);
    final_data(idx, :, :, :, 2) = reshape(V, [1, ny, nx, nz]);
    final_data(idx, :, :, :, 3) = reshape(W, [1, ny, nx, nz]);
    param_labels(idx,1:5) = [V_0, V_R, Delta_h, h_0, psi];
end

%% ===================== Save data =====================
addpath('npy-matlab-master\npy-matlab');

writeNPY(final_data,    'E:\shear-layer\shear_layer_wind_samples1000.npy');
writeNPY(param_labels, 'E:\shear-layer\shear_layer_wind_param1000.npy');

%% ===================== Verify output =====================
disp(['Data shape: ', mat2str(size(final_data))]);      % [1000,100,100,100,3]
disp(['Label shape: ', mat2str(size(param_labels))]);   % [1000,5]
