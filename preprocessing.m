%% 单人呼吸+心跳相位生命体征信号提取（只输出一幅时域图）
% 硬件：IWR6843ISK + DCA1000
% 数据：1024 帧，每帧 2 个 chirp，取每帧第 1 个 chirp，帧周期 50 ms => 慢时间采样率 20 Hz

clc; clear; close all;

%% ================== 基本参数 ==================
numADCSamples = 200;     % 每个 chirp 的 ADC 采样点数
numADCBits    = 16;      % ADC 位数
numTX         = 1;       % 发射天线数
numRX         = 4;       % 接收天线数
isReal        = 0;       % 0=复采样(I/Q)，1=实采样

% 雷达参数
Fs_adc   = 4e6;          % ADC 采样率 (Hz)
c        = 3e8;          % 光速 (m/s)
ts       = numADCSamples / Fs_adc;   % 一个 chirp 的采样时间
slope    = 77.006e12;    % 调频斜率 (Hz/s)
B_valid  = ts * slope;   % 有效带宽
deltaR0  = c / (2*B_valid);     % 原始距离分辨率(200点时)，约 4.6 cm
startFreq = 60e9;        % 起始频率 (Hz)
lambda    = c / startFreq;

% 慢时间采样率（生命体征信号）
framePeriod = 0.05;      % 每帧 50 ms（已知配置）
Fs_vital    = 1/framePeriod;   % 20 Hz

%% ================== 读取 bin 文件（相对路径） ==================
% 当前脚本所在目录，例如 ...\dataset_exp2024\1023data
curDir = fileparts(mfilename('fullpath'));

% bin 文件所在子目录：1023dataset\gby1023data
dataDir = fullfile(curDir, '1023dataset', 'gby1023data');

% 选择第几个 gby 数据：adc_1023gby1_Raw_0.bin
binIdx = 2;
fileNameOnly = sprintf('adc_1023gby%d_Raw_0.bin', binIdx);
Filename = fullfile(dataDir, fileNameOnly);

[fid, msg] = fopen(Filename, 'r');
if fid == -1
    error('无法打开数据文件：%s\n系统返回信息：%s', Filename, msg);
end
adcDataRow = fread(fid, 'int16');
fclose(fid);

% 若 ADC 位数不是 16 位，做符号扩展（这里按 16 位一般不需要）
if numADCBits ~= 16
    l_max = 2^(numADCBits-1)-1;
    adcDataRow(adcDataRow > l_max) = adcDataRow(adcDataRow > l_max) - 2^numADCBits;
end

%% ================== 数据重排为按 chirp 存放 ==================
fileSize = length(adcDataRow);     % int16 总长度

if isReal
    % 实采样：每个 chirp 的采样点数
    samplesPerChirp = numADCSamples * numRX;
else
    % 复采样：I/Q 两路
    samplesPerChirp = 2 * numADCSamples * numRX;
end

numChirps_total = floor(fileSize / samplesPerChirp);
fileSize        = numChirps_total * samplesPerChirp;    % 截断为整 chirp 长度
adcData         = adcDataRow(1:fileSize);

if isReal
    numChirps = fileSize / (numADCSamples * numRX);
    LVDS = reshape(adcData, numADCSamples*numRX, numChirps).';   % [numChirps, numADCSamples*numRX]
else
    % 复采样：组合 I/Q
    numChirps = fileSize / (2 * numADCSamples * numRX);
    LVDS = zeros(1, fileSize/2);
    counter = 1;
    for i = 1:4:fileSize-1   % 数据格式：I0,Q0,I1,Q1,...
        LVDS(counter)   = adcData(i)   + 1j*adcData(i+2);
        LVDS(counter+1) = adcData(i+1) + 1j*adcData(i+3);
        counter = counter + 2;
    end
    LVDS = reshape(LVDS, numADCSamples*numRX, numChirps).';      % [numChirps, numADCSamples*numRX]
end

%% ================== 取第 1 个接收通道，并只保留每帧第 1 个 chirp ==================
% 此时 numChirps = 2048（1024 帧 * 2 chirp）
adcAll = zeros(numRX, numChirps*numADCSamples);
for rx = 1:numRX
    for k = 1:numChirps
        adcAll(rx, (k-1)*numADCSamples+1 : k*numADCSamples) = ...
            LVDS(k, (rx-1)*numADCSamples+1 : rx*numADCSamples);
    end
end

% 只用第 1 个接收天线
retVal = reshape(adcAll(1,:), numADCSamples, numChirps);   % [200, 2048]

% 每帧 2 个 chirp，只取第 1 个 => 1024 个 chirp
numFrames   = numChirps/2;
process_adc = zeros(numADCSamples, numFrames);
for n = 1:2:numChirps
    process_adc(:, (n+1)/2) = retVal(:, n);
end
% 现在 process_adc: [200, 1024]，每列一个 chirp（慢时间采样率 20 Hz）

%% ================== 距离向 FFT，选择人体所在 range-bin ==================
RangFFT = 256;
adcdata = process_adc;            % [200, 1024]

fft_data = fft(adcdata, RangFFT, 1);  % 对距离维做 FFT，得到 [256, 1024]
fft_data = fft_data.';                % 转成 [1024, 256]，行=chirp，列=range-bin
fft_abs  = abs(fft_data);

% 距离分辨率（256 点零填充后的）
deltaR = Fs_adc * c / (2 * slope * RangFFT);

% 在 0.5~2.5 m 范围内做非相干积累，找能量最大 bin
range_energy = zeros(1, RangFFT);
range_max    = 0;
max_bin      = 1;

for r = 1:RangFFT
    dist_r = (r-1) * deltaR;
    if dist_r > 0.5 && dist_r < 2.5
        range_energy(r) = sum(fft_abs(:, r));
        if range_energy(r) > range_max
            range_max = range_energy(r);
            max_bin   = r;
        end
    end
end

fprintf('选中的人体距离 bin = %d, 约 %.2f m\n', max_bin, (max_bin-1)*deltaR);

%% ================== 提取该 range-bin 的相位，并做解缠 + 差分 + 平滑 ==================
% 原始相位（[-pi,pi]）
phase_raw = angle(fft_data(:, max_bin));    % 1024×1

% 相位解缠，得到连续相位
phase_unwrap = unwrap(phase_raw);

% 相位差分：抑制慢漂移，增强心跳；保留呼吸的变化
phase_diff = [0; diff(phase_unwrap)];       % 第一项补 0

% 滑动平均（窗口 5）去除脉冲噪声
phi = smoothdata(phase_diff, 'movmean', 5); % 生命体征相位信号（心率+呼吸）

%% ================== 构建真实时间轴并绘图 ==================
t = (0:length(phi)-1) / Fs_vital;   % 真实时间（秒）

figure;
plot(t, phi, 'LineWidth', 1.5);
grid on;
xlabel('Time (s)', 'FontSize', 12);
ylabel('Phase (rad)', 'FontSize', 12);
title(sprintf('Radar Vital-Sign Phase (Breathing + Heartbeat), gby%d', binIdx), 'FontSize', 14);

%%

%% ================== EEMD 分解：呼吸 / 心跳成分 ==================
% 这里使用整体生命体征相位信号 phi（差分+平滑后，心跳+呼吸）

Fs_vital = 1/0.05;          % 慢时间采样率 = 20 Hz（和前面保持一致）
phi_sig  = phi(:)';         % 转成行向量，方便部分 EEMD 函数

% ---- EEMD 参数（可以适当调节）----
NE        = 100;            % 集成次数（ensemble number）
noise_amp = 0.2*std(phi_sig); % 加噪声幅度，0.1~0.3*std 一般都可以
max_imf   = 10;             % 限制 IMF 个数，足够覆盖呼吸/心跳

% 调用 EEMD（根据你自己的 EEMD 工具箱接口，必要时调整参数顺序）
% 常见形式：imf = eemd(x, noise_amp, NE, max_imf);
imf = eemd(phi_sig, noise_amp, NE, max_imf);

% 如果你的 eemd 返回的是 IMF×N（每行一个 IMF），下面这句让它变成 N×IMF
if size(imf,1) < size(imf,2)
    imf = imf.';   % 现在 imf: [N 点数, K IMF 数]
end

[N,K] = size(imf);

% ---- 看每个 IMF 的主频，自动判断哪些是呼吸 / 心跳 ----
breath_idx = [];   % 呼吸 IMF 索引
heart_idx  = [];   % 心跳 IMF 索引

f_axis = (0:N-1)*(Fs_vital/N);  % 频率轴

for k = 1:K
    IMFk = imf(:,k);
    IMF_fft = abs(fft(IMFk));
    % 只看 0~一半频率（正频部分）
    IMF_fft_half = IMF_fft(1:floor(N/2));
    f_half       = f_axis(1:floor(N/2));

    [~, idx_max] = max(IMF_fft_half);
    f_peak = f_half(idx_max);   % 该 IMF 的主峰频率

    % 根据频段归类：
    % 呼吸：大约 0.1~0.5 Hz
    % 心跳：大约 0.8~2 Hz
    if f_peak >= 0.1 && f_peak <= 0.5
        breath_idx = [breath_idx, k];
    elseif f_peak >= 0.8 && f_peak <= 2
        heart_idx = [heart_idx, k];
    end
end

disp('EEMD 判定呼吸相关 IMF:');
disp(breath_idx);
disp('EEMD 判定心跳相关 IMF:');
disp(heart_idx);

% ---- 重构呼吸波 / 心跳波 ----
breath_wave = zeros(N,1);
heart_wave  = zeros(N,1);

if ~isempty(breath_idx)
    breath_wave = sum(imf(:, breath_idx), 2);
end
if ~isempty(heart_idx)
    heart_wave = sum(imf(:, heart_idx), 2);
end

% ---- 画图：原始生命体征 vs 呼吸 vs 心跳 ----
figure;
subplot(3,1,1);
plot(t, phi, 'k'); grid on;
xlabel('Time (s)');
ylabel('Phase (rad)');
title('Original Vital-Sign Phase (Breathing + Heartbeat)');

subplot(3,1,2);
plot(t, breath_wave, 'b'); grid on;
xlabel('Time (s)');
ylabel('Amplitude');
title('Breathing Component (EEMD-Reconstructed)');

subplot(3,1,3);
plot(t, heart_wave, 'r'); grid on;
xlabel('Time (s)');
ylabel('Amplitude');
title('Heartbeat Component (EEMD-Reconstructed)');
