clear;
close all;
clc;

%% Simulation Parameters
fault_time = 3/60;
line_length = 300;
fault_class = 0:11;
inception_angle = 0: 30 : 330;
fault_positions = 10:10:290;
fault_resistance = [0.001 1 10 50 100];

%% Load Simulation File
model = 'transmission_model_v1';
load_system(model);

%% Run Simulation
for fault = fault_class
    [FaultA, FaultB, FaultC, GroundFault] = fault_logic(fault);
    set_param([model '/Fault'], 'FaultA', FaultA, 'FaultB', FaultB, 'FaultC', FaultC, 'GroundFault', GroundFault);
    
    for angle = inception_angle
        F_time = fault_time + angle/(360*60);
        set_param([model '/Fault'], 'SwitchTimes', ['[' num2str(F_time) ' ' num2str(6/60) ']']);
        for distance = fault_positions
            L1 = distance;
            L2 = line_length - L1;
            set_param([model '/Line1'], 'Length', num2str(L1));
            set_param([model '/Line2'], 'Length', num2str(L2));
            for resistance = fault_resistance
                set_param([model '/Fault'], 'FaultResistance', num2str(resistance));
                try
                    output = sim(model);
                catch ME
                    warning ('FAILED: class_%g_angle_%g_dist_%g_res_%g\n%s',fault,angle,distance,resistance,ME.message);
                    continue;
                end

                signals = output.DATA.signals.values;
                time = output.DATA.time;

                folder = fullfile('../data/raw/',num2str(fault));
                if ~exist(folder,'dir')
                    mkdir(folder)
                end
                filename = fullfile(folder,sprintf('angle_%g_dist_%g_res_%g.mat',angle,L1,resistance));
                save(filename,'signals','time','angle','distance','resistance','fault');
            end
        end
    end
end

function [FaultA, FaultB, FaultC, GroundFault] = fault_logic(fault)
    FaultA = 'off'; FaultB = 'off'; FaultC = 'off'; GroundFault = 'off';
    switch fault
        case 0   % NF (No Fault)
        case 1   % AG
            FaultA = 'on'; GroundFault = 'on';            
        case 2   % BG
            FaultB = 'on'; GroundFault = 'on';            
        case 3   % CG
            FaultC = 'on'; GroundFault = 'on';            
        case 4   % AB
            FaultA = 'on'; FaultB = 'on';            
        case 5   % BC
            FaultB = 'on'; FaultC = 'on';            
        case 6   % CA
            FaultA = 'on'; FaultC = 'on';            
        case 7   % ABG
            FaultA = 'on'; FaultB = 'on'; GroundFault = 'on';
        case 8   % BCG
            FaultB = 'on'; FaultC = 'on'; GroundFault = 'on';            
        case 9  % CAG
            FaultA = 'on'; FaultC = 'on'; GroundFault = 'on';            
        case 10  % ABC
            FaultA = 'on'; FaultB = 'on'; FaultC = 'on';            
        case 11  % ABCG
            FaultA = 'on'; FaultB = 'on'; FaultC = 'on'; GroundFault = 'on';
    end
end
