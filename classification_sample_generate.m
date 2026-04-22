clc; clear; close all;
rng(42);

%% ===================== Configuration =====================
x = linspace(-500, 500, 100);
y = linspace(-500, 500, 100);
h = linspace(0, 500, 100);
[xw, yw, hw] = meshgrid(x, y, h);

[ny, nx, nz] = size(xw);

num_up_samples  = 100;
num_log_samples = 100;
num_fqb_samples = 100;
num_classes     = 3;
num_samples     = num_up_samples + num_log_samples + num_fqb_samples;

noise_amp = 0.1;

% Allocate memory, final shape: [N, y, x, h, 3]
final_data = zeros(num_samples, ny, nx, nz, 3, 'single');

%% ===================== Generate Latin hypercube parameters =====================
% Updraft parameters: [V_core, x0, y0, z0]
up_lhs = lhsdesign(num_up_samples, 4, 'criterion', 'maximin', 'iterations', 1000);
up_params = zeros(num_up_samples, 4);
up_params(:,1) = up_lhs(:,1) * 25 + 5;       % V_core: 5~30
up_params(:,2) = up_lhs(:,2) * 1000 - 500;   % x0: -500~500
up_params(:,3) = up_lhs(:,3) * 1000 - 500;   % y0: -500~500
up_params(:,4) = up_lhs(:,4) * 500;          % z0: 0~500

% Logarithmic wind parameters: [H_R, V_R, h0, psi]
log_lhs = lhsdesign(num_log_samples, 4, 'criterion', 'maximin', 'iterations', 1000);
log_params = zeros(num_log_samples, 4);
log_params(:,3) = max(log_lhs(:,3) * 2, 0.1);  % h0(void too small values): 0.1~2
log_params(:,1) = log_params(:,3) + 0.1 + (500 - log_params(:,3) - 0.1) .* log_lhs(:,1); % H_R: 0.1~500
log_params(:,2) = log_lhs(:,2) * 25 + 5;       % V_R: 5~30
log_params(:,4) = log_lhs(:,4) * 2 * pi;       % psi: 0~2pi

% Shear layer parameters: [V_0, V_R, Delta_h, h0, psi]
fqb_lhs = lhsdesign(num_fqb_samples, 5, 'criterion', 'maximin', 'iterations', 1000);
fqb_params = zeros(num_fqb_samples, 5);
fqb_params(:,1) = fqb_lhs(:,1) * 5 + 5;      % V_0: 5~10
fqb_params(:,2) = fqb_lhs(:,2) * 20 + 5;     % V_R: 5~25
fqb_params(:,3) = fqb_lhs(:,3) * 350 + 50;   % Delta_h: 50~400
fqb_params(:,4) = fqb_lhs(:,4) * 90 + 10;    % h0: 10~100
fqb_params(:,5) = fqb_lhs(:,5) * 2 * pi;     % psi: 0~2pi

%% ===================== Data generation =====================
idx = 0;

% ---------- Updraft ----------
R = 100;
k = 3;
for sample = 1:num_up_samples
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
end

% ---------- Logarithmic wind ----------
for sample = 1:num_log_samples
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
end

% ---------- Shear layer ----------
for sample = 1:num_fqb_samples
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
end

%% ===================== Classification labels =====================
class_labels = [ ...
    ones(num_up_samples,1); ...
    2*ones(num_log_samples,1); ...
    3*ones(num_fqb_samples,1) ];

onehot_labels = zeros(num_samples, num_classes, 'single');
onehot_labels(sub2ind([num_samples, num_classes], (1:num_samples)', class_labels)) = 1;

%% ===================== Save data =====================
addpath('npy-matlab-master\npy-matlab');

writeNPY(final_data,    'E:\classification\wind_samples_class.npy');
writeNPY(onehot_labels, 'E:\classification\wind_onehot_class.npy');

%% ===================== Verify output =====================
disp(['Data shape: ', mat2str(size(final_data))]);      % [300,100,100,100,3]
disp(['Label shape: ', mat2str(size(onehot_labels))]);  % [300,3]

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