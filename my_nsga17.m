function [sys, x0, str, ts] = my_nsga17(t, x, u, flag)
    persistent resultStorage;  % 持久化结果储存器
    persistent lastInputs;     % 持久化最后一次的输入

    % 定义用于存储结果的文件
    resultFile = 'resultStorage.mat';

    if isempty(resultStorage)
        if isfile(resultFile)
            % 如果文件存在，则从文件加载结果
            loadedData = load(resultFile);
            resultStorage = loadedData.resultStorage;
        else
            % 如果文件不存在，则初始化空的 containers.Map
            resultStorage = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end
    end

    if isempty(lastInputs)
        lastInputs = [];
    end

    switch flag
        case 0
            [sys, x0, str, ts] = mdlInitializeSizes;
        case 3
            key = generateKey(u);  % 生成哈希表的键值
            if isKey(resultStorage, key)
                sys = resultStorage(key);  % 如果已存在，直接返回存储的结果
            else
                sys = mdlOutputs(t, x, u);
                resultStorage(key) = sys;  % 存储计算结果

                % 将结果存储到文件
                save(resultFile, 'resultStorage');
            end
            lastInputs = u;  % 更新最后一次的输入
        case {1, 2, 4, 9}
            sys = [];
        otherwise
            error(['未处理的标志 = ', num2str(flag)]);
    end
end

function key = generateKey(u)
    % 将输入向量转化为字符串，用于作为哈希表的键
    key = mat2str(u);
end

function [sys, x0, str, ts] = mdlInitializeSizes
    sizes = simsizes;
    sizes.NumContStates = 0;
    sizes.NumDiscStates = 0;
    sizes.NumOutputs = 4;
    sizes.NumInputs = 15;  % 更新为15个输入（新增转向角输入）
    sizes.DirFeedthrough = 1;
    sizes.NumSampleTimes = 1;

    sys = simsizes(sizes);
    x0 = [];
    str = [];
    ts = [0.01, 0];  % 步长为0.01
end

function sys = mdlOutputs(t, x, u)
    % 定义参数
    Rw = 0.325;
    Iw = 1.06;
    f = 0;
    populationSize = 220;
    maxGenerations = 50;
    mutationRate = 0.2;
    L = 2.91;        % 轴距
    W = 1.675;       % 轮距

    % 输入变量
    FZ11 = u(1); FZ12 = u(2); FZ21 = u(3); FZ22 = u(4);
    Fx11 = u(5); Fx12 = u(6); Fx21 = u(7); Fx22 = u(8);
    TM = u(9); Vx = u(10);
    mu11 = u(11); mu12 = u(12); mu21 = u(13); mu22 = u(14);
    delta = u(15);  % 新增转向角输入

    % 目标函数定义
    objectiveFunction = @(T) computeObjectives(T, u, Rw, Iw, f, Vx, TM, max(0.1, t), ...
        [mu11, mu12, mu21, mu22], L, W, delta);

    % 初始化种群 - 现在允许负值
    population = initializePopulation(populationSize, TM);

    for generation = 1:maxGenerations
        % 评估种群适应度
        fitness = evaluatePopulation(population, objectiveFunction);
        
        % 执行非支配排序
        [fronts, crowdingDistances] = nonDominatedSortAndCrowding(fitness);
        
        % 选择父代
        parents = selectNSGA2Parents(population, fronts, crowdingDistances, populationSize);
        
        % 交叉与变异生成子代
        offspring = crossoverAndMutate(parents, mutationRate, TM);
        
        % 合并父代和子代
        population = [population; offspring];
        
        % 再次评估适应度并进行选择，保留精英
        fitness = evaluatePopulation(population, objectiveFunction);
        [fronts, crowdingDistances] = nonDominatedSortAndCrowding(fitness);
        population = selectNSGA2Parents(population, fronts, crowdingDistances, populationSize);
    end

    bestIndividual = selectBestIndividual(population, objectiveFunction);
    T_opt = bestIndividual;

    % 调整转矩以满足附着系数限制条件（考虑负值）
    T_adjusted = adjustTorques(T_opt, TM, [mu11, mu12, mu21, mu22], [FZ11, FZ12, FZ21, FZ22], Rw, f);

    % 将调整后的转矩作为输出
    sys = T_adjusted';
end

function population = initializePopulation(populationSize, TM)
    % 初始化种群并允许负值，但总和仍等于 TM
    if populationSize <= 0 || mod(populationSize, 1) ~= 0
        error('populationSize 必须是正整数标量');
    end
    
    % 随机生成驱动力矩个体（范围扩大，允许负值）
    population = 2 * rand(populationSize, 4) - 1;  % 范围[-1,1]
    
    % 调整种群中的每个个体，使得驱动力矩总和为 TM
    for i = 1:populationSize
        % 归一化当前个体，使其总和为 TM
        population(i, :) = population(i, :) / sum(population(i, :)) * TM;
    end
end

function T_adjusted = adjustTorques(T, TM, mu, FZ, Rw, f)
    % 调整转矩以满足附着系数约束（考虑负值）
    T_sum = sum(T);
    if abs(T_sum) < 1e-6  % 防止除以零
        T_adjusted = zeros(1,4);
        return;
    end
    
    % 计算每个车轮的最大允许转矩（考虑正负）
    maxTorques = [(mu(1)+f) * FZ(1) * Rw, (mu(2)+f) * FZ(2) * Rw, ...
                 (mu(3)+f) * FZ(3) * Rw, (mu(4)+f) * FZ(4) * Rw];
    
    % 计算每个车轮的最小允许转矩（制动情况）
    minTorques = -maxTorques;
    
    % 首先按比例分配
    T_adjusted = T / T_sum * TM;
    
    % 然后应用约束
    for i = 1:4
        if T_adjusted(i) > 0
            T_adjusted(i) = min(T_adjusted(i), maxTorques(i));
        else
            T_adjusted(i) = max(T_adjusted(i), minTorques(i));
        end
    end
    
    % 确保总和仍然等于TM（可能需要二次调整）
    if abs(sum(T_adjusted) - TM) > 1e-6
        remaining = TM - sum(T_adjusted);
        % 按比例分配剩余转矩到未饱和的车轮
        available = ones(1,4);
        for i = 1:4
            if (T_adjusted(i) >= maxTorques(i) && remaining > 0) || ...
               (T_adjusted(i) <= minTorques(i) && remaining < 0)
                available(i) = 0;
            end
        end
        if sum(available) > 0
            T_adjusted = T_adjusted + remaining * available / sum(available);
            % 再次应用约束
            for i = 1:4
                if T_adjusted(i) > 0
                    T_adjusted(i) = min(T_adjusted(i), maxTorques(i));
                else
                    T_adjusted(i) = max(T_adjusted(i), minTorques(i));
                end
            end
        end
    end
end

function fitness = evaluatePopulation(population, objectiveFunction)
    % 评估种群适应度
    numIndividuals = size(population, 1);
    fitness = zeros(numIndividuals, 5);  % 现在有5个目标

    for i = 1:numIndividuals
        fitness(i, :) = objectiveFunction(population(i, :));
    end
end

function [fronts, crowdingDistances] = nonDominatedSortAndCrowding(fitness)
    % 非支配排序及拥挤距离计算
    numIndividuals = size(fitness, 1);
    dominationCount = zeros(numIndividuals, 1);
    dominatedSet = cell(numIndividuals, 1);
    fronts = cell(1, 1);
    rank = zeros(numIndividuals, 1);

    % 初始化非支配排序
    front1 = [];
    for i = 1:numIndividuals
        for j = 1:numIndividuals
            if all(fitness(i, :) <= fitness(j, :)) && any(fitness(i, :) < fitness(j, :))
                dominatedSet{i} = [dominatedSet{i}, j];
            elseif all(fitness(j, :) <= fitness(i, :)) && any(fitness(j, :) < fitness(i, :))
                dominationCount(i) = dominationCount(i) + 1;
            end
        end
        if dominationCount(i) == 0
            rank(i) = 1;
            front1 = [front1, i];
        end
    end

    fronts{1} = front1;
    currentFront = 1;
    while ~isempty(fronts{currentFront})
        nextFront = [];
        for i = fronts{currentFront}
            for j = dominatedSet{i}
                dominationCount(j) = dominationCount(j) - 1;
                if dominationCount(j) == 0
                    rank(j) = currentFront + 1;
                    nextFront = [nextFront, j];
                end
            end
        end
        currentFront = currentFront + 1;
        fronts{currentFront} = nextFront;
    end

    % 计算拥挤度距离
    numObjectives = size(fitness, 2);
    crowdingDistances = zeros(numIndividuals, 1);

    for f = 1:length(fronts)
        front = fronts{f};
        if isempty(front)
            continue;
        end
        numFrontIndividuals = length(front);
        for m = 1:numObjectives
            [~, sortIdx] = sort(fitness(front, m));
            crowdingDistances(front(sortIdx(1))) = inf;
            crowdingDistances(front(sortIdx(end))) = inf;
            for i = 2:numFrontIndividuals - 1
                crowdingDistances(front(sortIdx(i))) = crowdingDistances(front(sortIdx(i))) + ...
                    (fitness(front(sortIdx(i + 1)), m) - fitness(front(sortIdx(i - 1)), m)) / ...
                    (max(fitness(:, m)) - min(fitness(:, m)));
            end
        end
    end
end

function parents = selectNSGA2Parents(population, fronts, crowdingDistances, populationSize)
    % 根据非支配排序和拥挤距离选择父代
    numIndividuals = size(population, 1);
    selected = [];
    for f = 1:length(fronts)
        front = fronts{f};
        if isempty(front)
            continue;
        end
        if length(selected) + length(front) <= populationSize
            selected = [selected, front];
        else
            [~, sortIdx] = sort(crowdingDistances(front), 'descend');
            selected = [selected, front(sortIdx(1:(populationSize - length(selected))))];
            break;
        end
    end
    parents = population(selected, :);
end

function offspring = crossoverAndMutate(parents, mutationRate, TM)
    % 交叉与变异生成子代（允许负值）
    numParents = size(parents, 1);
    offspring = zeros(numParents, size(parents, 2));

    for i = 1:2:numParents
        parent1 = parents(i, :);
        parent2 = parents(min(i + 1, numParents), :);
        alpha = rand;  % 均匀交叉
        child1 = alpha * parent1 + (1 - alpha) * parent2;
        child2 = alpha * parent2 + (1 - alpha) * parent1;

        % 变异（允许负值）
        if rand < mutationRate
            mutationPoint = randi(length(child1));
            child1(mutationPoint) = child1(mutationPoint) + 0.2 * (2*rand-1); % [-0.2,0.2]范围
        end
        if rand < mutationRate
            mutationPoint = randi(length(child2));
            child2(mutationPoint) = child2(mutationPoint) + 0.2 * (2*rand-1);
        end

        % 确保总和为TM
        child1 = child1 / sum(child1) * TM;
        child2 = child2 / sum(child2) * TM;

        offspring(i, :) = child1;
        if i + 1 <= numParents
            offspring(i + 1, :) = child2;
        end
    end
end

function bestIndividual = selectBestIndividual(population, objectiveFunction)
    % 选择最佳个体（适应度最优的个体）
    fitness = evaluatePopulation(population, objectiveFunction);
    [~, bestIdx] = min(sum(fitness, 2));  % 假设适应度是目标函数值的和
    bestIndividual = population(bestIdx, :);
end

function objectives = computeObjectives(T, u, Rw, Iw, f, Vx, TM, t, mu, L, W, delta)
    % 确保 T 的总和非零
    if abs(sum(T)) < 1e-6
        error('T 的总和为零，不能进行调整');
    end

    % 调整 T 以满足附着系数约束条件（考虑负值）
    T_sum = sum(T);
    maxTorques = [(mu(1)+f) * u(1) * Rw, (mu(2)+f) * u(2) * Rw, ...
                 (mu(3)+f) * u(3) * Rw, (mu(4)+f) * u(4) * Rw];
    minTorques = -maxTorques;
    
    T_adjusted = T / T_sum * TM;
    for i = 1:4
        if T_adjusted(i) > 0
            T_adjusted(i) = min(T_adjusted(i), maxTorques(i));
        else
            T_adjusted(i) = max(T_adjusted(i), minTorques(i));
        end
    end

    % 计算角加速度（考虑转矩方向）
    W11_dt = 2*pi/60*(T_adjusted(1) - u(5) * Rw - u(1) * f * Rw) / Iw;
    W12_dt = 2*pi/60*(T_adjusted(2) - u(6) * Rw - u(2) * f * Rw) / Iw;
    W21_dt = 2*pi/60*(T_adjusted(3) - u(7) * Rw - u(3) * f * Rw) / Iw;
    W22_dt = 2*pi/60*(T_adjusted(4) - u(8) * Rw - u(4) * f * Rw) / Iw;

    stepSize = 0.01;
    time_vector = 0:stepSize:max(0.5, t); 

    if length(time_vector) < 2
        error('时间向量的长度不足。请增加时间范围 t 或减少步长');
    end

    % 对角加速度进行积分以计算角速度（考虑初始速度）
    W11 = 0.85 + cumtrapz(time_vector, W11_dt * ones(size(time_vector)));
    W12 = 0.85 + cumtrapz(time_vector, W12_dt * ones(size(time_vector)));
    W21 = 0.85 + cumtrapz(time_vector, W21_dt * ones(size(time_vector)));
    W22 = 0.85 + cumtrapz(time_vector, W22_dt * ones(size(time_vector)));

    % 计算每个车轮的线速度
    V11 = Rw * W11(end);
    V12 = Rw * W12(end);
    V21 = Rw * W21(end);
    V22 = Rw * W22(end);

    % 计算滑转率（考虑负值情况）
    n11 = (V11 - Vx) / max(abs(V11), 0.1);  % 避免除以零
    n12 = (V12 - Vx) / max(abs(V12), 0.1);
    n21 = (V21 - Vx) / max(abs(V21), 0.1);
    n22 = (V22 - Vx) / max(abs(V22), 0.1);

    % 计算载荷分配比例
    loadDistribution = [u(1), u(2), u(3), u(4)];
    totalLoad = sum(loadDistribution);
    loadRatio = loadDistribution / totalLoad;

    % 计算滑转率方差 f2
    slipRatios = [n11, n12, n21, n22];
    f2 = var(slipRatios);

    % 计算车轮速度方差 f3（转向时失效）
    wheelSpeeds = [V11, V12, V21, V22];
    if abs(delta) > 0.002  % 转向角超过阈值时，f3失效（设为0）
        f3 = 0;
    else
        f3 = var(wheelSpeeds);
    end

    % 计算所有车轮滑转率的总和 f4
    f4 = sum(abs(slipRatios));  % 使用绝对值确保一致性

    % 计算驱动转矩比例与载荷比例的差异平方和 f1
    T_ratio = T_adjusted / sum(abs(T_adjusted));  % 使用绝对值比例
    f1 = sum((T_ratio - loadRatio).^2);

    % 第五个目标函数：基于阿克曼转向模型的速度差
    % 计算转弯半径（取delta的绝对值以确保半径为正）
    if abs(delta) < 0.002  % 直行情况
        R = inf;
    else
        R = L / tan(abs(delta));  % 取绝对值确保半径为正
    end
    
    % 计算各车轮的理想速度（阿克曼转向模型）
    if isinf(R)  % 直行情况
        V_ideal11 = Vx;
        V_ideal12 = Vx;
        V_ideal21 = Vx;
        V_ideal22 = Vx;
    else
        % 根据转向方向调整内外轮计算
        if delta > 0  % 左转
            % 内轮（左轮）和外轮（右轮）
            R_inner_front = sqrt((R - W/2)^2 + L^2);
            R_outer_front = sqrt((R + W/2)^2 + L^2);
            R_inner_rear = R - W/2;
            R_outer_rear = R + W/2;
            
            V_ideal11 = Vx * R_inner_front / R;
            V_ideal12 = Vx * R_outer_front / R;
            V_ideal21 = Vx * R_inner_rear / R;
            V_ideal22 = Vx * R_outer_rear / R;
        else  % 右转
            % 内轮（右轮）和外轮（左轮）
            R_inner_front = sqrt((R + W/2)^2 + L^2);
            R_outer_front = sqrt((R - W/2)^2 + L^2);
            R_inner_rear = R + W/2;
            R_outer_rear = R - W/2;
            
            V_ideal11 = Vx * R_outer_front / R;
            V_ideal12 = Vx * R_inner_front / R;
            V_ideal21 = Vx * R_outer_rear / R;
            V_ideal22 = Vx * R_inner_rear / R;
        end
    end
    
    % 计算实际速度与理想速度的平方差
    f5 = (V11 - V_ideal11)^2 + (V12 - V_ideal12)^2 + ...
         (V21 - V_ideal21)^2 + (V22 - V_ideal22)^2;

    objectives = [f1, f2, f3, f4, f5];
end