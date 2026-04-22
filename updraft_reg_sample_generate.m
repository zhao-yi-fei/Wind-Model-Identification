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
% Updraft parameters: [V_core, x0, y0, z0]
up_lhs = lhsdesign(num_samples, 4, 'criterion', 'maximin', 'iterations', 1000);
up_params = zeros(num_samples, 4);
up_params(:,1) = up_lhs(:,1) * 25 + 5;       % V_core: 5~30
up_params(:,2) = up_lhs(:,2) * 1000 - 500;   % x0: -500~500
up_params(:,3) = up_lhs(:,3) * 1000 - 500;   % y0: -500~500
up_params(:,4) = up_lhs(:,4) * 500;          % z0: 0~500


%% ===================== Data generation =====================
idx = 0;

R = 100;
k = 3;
for sample = 1:num_samples
    idx = idx + 1;

    V_core = up_params(sample,1);
    x_0    = up_params(sample,2);
    y_0    = up_params(sample,3);
    z_0    = up_params(sample,4);

    [U, V, W] = updraft(xw, yw, hw, V_core, R, k, x_0, y_0, z_0);
    
    % add noise
    % U = single(U + (rand(size(U))*2 - 1) * noise_amp);
    % V = single(V + (rand(size(V))*2 - 1) * noise_amp);
    % W = single(W + (rand(size(W))*2 - 1) * noise_amp);

    final_data(idx, :, :, :, 1) = reshape(U, [1, ny, nx, nz]);
    final_data(idx, :, :, :, 2) = reshape(V, [1, ny, nx, nz]);
    final_data(idx, :, :, :, 3) = reshape(W, [1, ny, nx, nz]);
    param_labels(idx,1:4) = [V_core,x_0, y_0,z_0];
end

%% ===================== Save data =====================
addpath('npy-matlab-master\npy-matlab');

writeNPY(final_data,    'E:\updraft\updraft_wind_samples1000.npy');
writeNPY(param_labels, 'E:\updraft\updraft_wind_param1000.npy');

%% ===================== Verify output =====================
disp(['Data shape: ', mat2str(size(final_data))]);      % [1000,100,100,100,3]
disp(['Label shape: ', mat2str(size(param_labels))]);   % [1000,4]

%% ===================== Updraft function =====================
function [U, V, W] = updraft(xw, yw, hw, V_core, R, k, x0, y0, z0)
    d = sqrt((xw - x0).^2 + (yw - y0).^2);
    epsd = 1e-8 * max(R, max(d(:)) + eps);

    inZ  = abs(hw - z0) <= k * R + epsd;
    inR  = d < R - epsd;
    onAx = d <= epsd;
    onEd = abs(d - R) <= epsd;

    W = zeros(size(d));
    U = zeros(size(d));
    V = zeros(size(d));

    % Interior region：0 < d < R
    mask_int = (~onAx) & inR & inZ;
    W(mask_int) = V_core .* R ./ (2*pi*d(mask_int)) .* ...
                  sin(pi*d(mask_int)/R) .* ...
                  (cos(pi*(hw(mask_int)-z0)/(k*R)) + 1);

    % Axis: limiting value as d -> 0
    mask_axis = onAx & inZ;
    W(mask_axis) = V_core;

    % U and V are only defined in the interior region
    U(mask_int) = -W(mask_int) .* (hw(mask_int)-z0) .* (xw(mask_int)-x0) ./ ...
                  (d(mask_int) .* (d(mask_int)-R) * k^2);

    V(mask_int) = -W(mask_int) .* (hw(mask_int)-z0) .* (yw(mask_int)-y0) ./ ...
                  (d(mask_int) .* (d(mask_int)-R) * k^2);

    % Boundary d ≈ R: set to zero for numerical robustness
    W(onEd & inZ) = 0;
    U(onEd & inZ) = 0;
    V(onEd & inZ) = 0;

    % Outside the valid region: set to zero
    mask_out = (~inR & ~onAx & ~onEd) | (~inZ);
    W(mask_out) = 0;
    U(mask_out) = 0;
    V(mask_out) = 0;
end