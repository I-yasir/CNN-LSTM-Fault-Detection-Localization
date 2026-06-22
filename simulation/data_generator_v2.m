%% Transmission Line Fault Dataset Generation Script (Optimized Batch Mode)
% Compatible with MATLAB 2022b and later
% Generates simulation runs in batches and saves directly to HDF5
clear; clk = tic;
model_name = 'transmission_model_v2'; % Your exact Simulink file name

%% 1. Dataset Configuration & Balancing
num_classes = 11;
runs_per_class = 1500; % Pure (60%), Harm (15%), Noise (15%), Both (10%)
n_pure  = round(runs_per_class * 0.60);
n_harm  = round(runs_per_class * 0.15);
n_noise = round(runs_per_class * 0.15);
n_both  = runs_per_class - n_pure - n_harm - n_noise;
total_runs = num_classes * runs_per_class;

% Fault Class Mapping Matrix: [FaultA, FaultB, FaultC, GroundFault]
fault_modes = [
    0, 0, 0, 0; % 1: NF (No Fault)
    1, 0, 0, 1; % 2: AG
    0, 1, 0, 1; % 3: BG
    0, 0, 1, 1; % 4: CG
    1, 1, 0, 0; % 5: AB
    0, 1, 1, 0; % 6: BC
    1, 0, 1, 0; % 7: AC
    1, 1, 0, 1; % 8: ABG
    0, 1, 1, 1; % 9: BCG
    1, 0, 1, 1; % 10: ACG
    1, 1, 1, 1  % 11: ABCG
];

% Pre-allocate deterministic balanced configuration structures
rng(42); % For reproducibility
config(total_runs) = struct();
idx = 1;
for c = 1:num_classes
    for scenario_type = 1:4
        if scenario_type == 1, n_type = n_pure;
        elseif scenario_type == 2, n_type = n_harm;
        elseif scenario_type == 3, n_type = n_noise;
        else, n_type = n_both;
        end
        for k = 1:n_type
            config(idx).class = c;
            config(idx).scenario = scenario_type;
            config(idx).angle = rand() * 360;
            config(idx).Rfault = 10^(log10(0.01) + rand()*(log10(100)-log10(0.01)));
            config(idx).location = 5 + (95 - 5) * rand();
            idx = idx + 1;
        end
    end
end

% Shuffle configurations
config = config(randperm(total_runs));

%% 2. HDF5 File Initialization
simualtionDir = pwd;
dataPath = fullfile(simulationDir, '..' , 'data');
if ~exist (dataPath, 'dir')
    mkdir(dataPath)
end
hdf5_file = fullfile(dataPath, 'transmission_fault_dataset.h5');
if exist(hdf5_file, 'file')
    delete(hdf5_file);
    fprintf('Existing HDF5 file deleted: %s\n', hdf5_file);
else
    fprintf('Creating new HDF5 file: %s\n', hdf5_file);
end
num_time_steps = 2001; 
num_channels = 6; % 3-Phase Voltage + 3-Phase Current

h5create(hdf5_file, '/signals', [num_channels, num_time_steps, total_runs], ...
         'ChunkSize', [num_channels, num_time_steps, 1], 'Datatype', 'single');
h5create(hdf5_file, '/fault_class', [total_runs, 1], 'Datatype', 'int32');
h5create(hdf5_file, '/fault_location', [total_runs, 1], 'Datatype', 'single');

%% 3. Build SimulationInput Array
fprintf('Loading model and preparing simulation configurations...\n');
load_system(model_name);
f = 60; 
snr_options = [50, 40, 30, 20];

% Preallocate the SimulationInput array
simIn(1:total_runs) = Simulink.SimulationInput(model_name);
for i = 1:total_runs
    c_idx = config(i).class;
    scen = config(i).scenario;
    loc_pct = config(i).location;
    r_fault = config(i).Rfault;
    ang = config(i).angle;
    
    L1 = 300 * (loc_pct / 100);
    L2 = 300 * (1 - (loc_pct / 100));
    t_inception = 3/f + (ang / 360) * (1 / f);
    
    % Apply line parameters
    simIn(i) = simIn(i).setBlockParameter([model_name '/Line1'], 'Length', num2str(L1));
    simIn(i) = simIn(i).setBlockParameter([model_name '/Line2'], 'Length', num2str(L2));
    
    % Configure Fault Block
    faultPath = [model_name '/Fault'];
    if c_idx == 1
        simIn(i) = simIn(i).setBlockParameter(faultPath, 'FaultA', 'off');
        simIn(i) = simIn(i).setBlockParameter(faultPath, 'FaultB', 'off');
        simIn(i) = simIn(i).setBlockParameter(faultPath, 'FaultC', 'off');
        simIn(i) = simIn(i).setBlockParameter(faultPath, 'GroundFault', 'off');
    else
        simIn(i) = simIn(i).setBlockParameter(faultPath, 'FaultA',          ifThenElse(fault_modes(c_idx, 1), 'on', 'off'));
        simIn(i) = simIn(i).setBlockParameter(faultPath, 'FaultB',          ifThenElse(fault_modes(c_idx, 2), 'on', 'off'));
        simIn(i) = simIn(i).setBlockParameter(faultPath, 'FaultC',          ifThenElse(fault_modes(c_idx, 3), 'on', 'off'));
        simIn(i) = simIn(i).setBlockParameter(faultPath, 'GroundFault',     ifThenElse(fault_modes(c_idx, 4), 'on', 'off'));
        simIn(i) = simIn(i).setBlockParameter(faultPath, 'FaultResistance', num2str(r_fault));
        simIn(i) = simIn(i).setBlockParameter(faultPath, 'SwitchTimes',     ['[', num2str(t_inception), ' 1.0]']); 
    end
    
    %% Inject Harmonics (Scenario 2 and 4)
    src1_path = [model_name '/Generation/Source'];
    src2_path = [model_name '/Generation/Source1'];
    
    simIn(i) = simIn(i).setBlockParameter(src1_path, 'HarmonicGeneration', 'off');
    
    if scen == 2 || scen == 4
        harmonic3 = [3 0.01+0.04*rand() rand()*360 0];
        harmonic5 = [5 0.01+0.04*rand() rand()*360 2];
        harmonic7 = [7 0.01+0.04*rand() rand()*360 1];
        harmonic11 = [11 0.01+0.04*rand() rand()*360 2];
        
        simIn(i) = simIn(i).setBlockParameter(src1_path, 'HarmonicGeneration', 'on');
        simIn(i) = simIn(i).setBlockParameter(src1_path, 'HarmonicA', mat2str(harmonic3), ...
                                              src1_path, 'HarmonicB', mat2str(harmonic5));
                                          
        simIn(i) = simIn(i).setBlockParameter(src2_path, 'HarmonicA', mat2str(harmonic7), ...
                                              src2_path, 'HarmonicB', mat2str(harmonic11));
    end
    % NOTE: Callback function assignment removed. Section 4 now handles data saving smoothly.
end

%% 4. Execution in Batches (Optimized for Speed and RAM)
fprintf('Starting %d optimized simulation runs in batches...\n', total_runs);

batch_size = 100; % Reduced to 100 for ideal operational memory optimization
num_batches = ceil(total_runs / batch_size);

for b = 1:num_batches
    start_idx = (b-1) * batch_size + 1;
    end_idx = min(b * batch_size, total_runs);
    current_batch_size = end_idx - start_idx + 1;
    
    % Slice configurations for this batch
    simIn_batch = simIn(start_idx:end_idx);
    
    % Run the small batch
    sim_outputs = sim(simIn_batch, 'ShowProgress', 'off'); 
    
    % Preallocate temporary batch buffers in RAM
    batch_signals = zeros(num_channels, num_time_steps, current_batch_size, 'single');
    batch_classes = zeros(current_batch_size, 1, 'int32');
    batch_locations = zeros(current_batch_size, 1, 'single');
    
    % Process batch results
    for idx_in_batch = 1:current_batch_size
        global_idx = start_idx + idx_in_batch - 1;
        c_idx = config(global_idx).class;
        scen = config(global_idx).scenario;
        loc_pct = config(global_idx).location;
        
        try
            current_out = sim_outputs(idx_in_batch);
            
            % Check if Simulink itself logged an execution error message
            if ~isempty(current_out.ErrorMessage)
                error('Simulink:RunError', 'Simulink Core Error: %s', current_out.ErrorMessage);
            end
            
            % Confirm existence of variable inside the output
            if ~isprop(current_out, 'simout') && ~any(strcmp(current_out.who, 'simout'))
                error('Simulink:VariableMissing', 'Could not find "simout". Exported elements: %s', strjoin(current_out.who, ', '));
            end
            
            raw_signals = current_out.simout.signals.values; % Shape: Time x Channels
            
            % Apply Noise Injection
            if scen == 3 || scen == 4
                selected_snr = snr_options(randi(4));
                for ch = 1:num_channels
                    raw_signals(:, ch) = awgn(raw_signals(:, ch), selected_snr, 'measured');
                end
            end
            
            % Transpose matrix to format: Channels x Time
            batch_signals(:, :, idx_in_batch) = raw_signals';
            
        catch ME
            fprintf('\n[ERROR] Simulation %d failed. Details: %s\n', global_idx, ME.message);
            batch_signals(:, :, idx_in_batch) = NaN;
        end
        
        % Store target annotations
        batch_classes(idx_in_batch) = int32(c_idx - 1); % 0-indexed for ML frameworks
        if c_idx == 1
            batch_locations(idx_in_batch) = single(-1);
        else
            batch_locations(idx_in_batch) = single(loc_pct);
        end
    end
    
    % Write the entire batch chunk directly to the HDF5 file
    h5write(hdf5_file, '/signals', batch_signals, ...
            [1, 1, start_idx], [num_channels, num_time_steps, current_batch_size]);
        
    h5write(hdf5_file, '/fault_class', batch_classes, ...
            [start_idx, 1], [current_batch_size, 1]);
        
    h5write(hdf5_file, '/fault_location', batch_locations, ...
            [start_idx, 1], [current_batch_size, 1]);
    
    % Clear heavy batch allocations to reset RAM footprint cleanly
    clear sim_outputs batch_signals batch_classes batch_locations;
    
    % Progress tracking log
    fprintf('Completed Batch %d/%d (Runs %d-%d). Total Elapsed Time: %.2f mins\n', ...
            b, num_batches, start_idx, end_idx, toc(clk)/60);
end

fprintf('\nDataset Generation Complete! File saved as: %s\n', hdf5_file);

%% Helper Logic Functions
function out = ifThenElse(condition, trueVal, falseVal)
    if condition, out = trueVal; else, out = falseVal; end
end