function imf = eemd(x, noise_amp, NE, max_imf)
% 依赖：无
% 输入：
%   x         — 输入信号（行或列向量）
%   noise_amp — 加噪标准差，例如 0.2*std(x)
%   NE        — 集成次数（常用 50–100）
%   max_imf   — 最多输出的 IMF 个数
%
% 输出：
%   imf       — 每列一个 IMF，大小为 [length(x), max_imf]

x = x(:);              % 转成列向量
N = length(x);

all_imf = zeros(N, max_imf);   % 用于 ensemble 平均

for n = 1:NE
    % 添加白噪声
    w = noise_amp * randn(N,1);
    x_noisy = x + w;

    % 使用 EMD 分解（MATLAB 内置）
    imf_noisy = emd(x_noisy, 'MaxNumIMF', max_imf);

    % IMF 逐列累加
    for k = 1:size(imf_noisy,2)
        all_imf(:,k) = all_imf(:,k) + imf_noisy(:,k);
    end
end

% 求 ensemble 平均
imf = all_imf / NE;
end
