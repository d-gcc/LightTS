import copy, torch, time
from torch import nn
from pathlib import Path
from utils.data import get_loaders
from utils.inception import InceptionModel
from utils.distiller import *
from utils.trainer import train_single, train_distilled, validation, evaluate, evaluate_ensemble
from utils.cawpe import train_probabilities

                
def BasicStudent(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_loader, val_loader, test_loader = get_loaders(config)
    start_training = time.time()
    
    for epoch in range(1, config.epochs + 1):
        train_single(epoch, train_loader, model, optimizer, config)
    
        if (epoch) % 100 == 0 or config.pid == 0:
            training_time = time.time() - start_training
            current_accuracy = evaluate(test_loader, model, config, epoch, training_time)
            
def BaselineStudent(model, config):
    teachers = [i for i in range(0,config.teachers)]

    model_s = model
    model_s.eval()
    model_s = model_s.to(config.device)
    optimizer = torch.optim.Adam(model_s.parameters(), lr=config.lr)
    
    module_list = nn.ModuleList([])
    module_list.append(model_s)

    criterion_list = nn.ModuleList([])
    
    if config.num_classes > 1:
        criterion_list.append(nn.CrossEntropyLoss())
    else:
        criterion_list.append(nn.BCEWithLogitsLoss())
    
    if config.evaluation == 'classic':
        if config.num_classes > 1:
            criterion_list.append(KDEnsemble(config.kd_temperature, config.device))
        else:
            criterion_list.append(KDEnsembleOne(config.kd_temperature, config.device))
        teacher_probs = 0
    elif config.evaluation == 'agree':
        if config.num_classes > 1:
            criterion_list.append(DistillKL(config.kd_temperature))
        else:
            criterion_list.append(DistillKLOne(config.kd_temperature))
        teacher_probs = 0
    elif config.evaluation == 'cawpe':
        if config.num_classes > 1:
            criterion_list.append(DistillKL(config.kd_temperature))
        else:
            criterion_list.append(DistillKLOne(config.kd_temperature))
        teacher_probs = train_probabilities(config)     
    elif config.evaluation == 'reinforced':
        if config.num_classes > 1:
            criterion_list.append(DistillKL(config.kd_temperature))
        else:
            criterion_list.append(DistillKLOne(config.kd_temperature))
        teacher_probs = torch.full((1,config.teachers), 1/config.teachers, dtype=torch.float32, 
                                   device = config.device).squeeze()
        
    if config.teacher_type == 'Inception':
        train_loader, val_loader, test_loader = get_loaders(config)
    else:
        config.batch_size = 10000
        train_loader, val_loader, test_loader = get_loaders(config)
        
    # Teachers
    teacher_list = []
    teacher_val_list = []
    if config.teacher_type == 'Inception':
        for teacher in teachers:
            savepath = Path('./teachers/Inception_' + config.dataset + '_' + str(teacher) + '_teacher.pkl')
            teacher_config = copy.deepcopy(config)
            teacher_config.bit1 = teacher_config.bit2 = teacher_config.bit3 = 32
            teacher_config.layer1 = teacher_config.layer2 = teacher_config.layer3 = 3
            model_t = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                           bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                           num_pred_classes=config.num_classes,config=teacher_config)

            model_t.load_state_dict(torch.load(savepath, map_location=config.device))
            model_t.eval()
            model_t = model_t.to(config.device)
            module_list.append(model_t)
    else:
        for teacher in teachers:
            savepath = Path('./teachers/'+ config.teacher_type + '_' + config.dataset + '_' + str(teacher) + '_teacher.pkl')
            with open(savepath,'rb') as file:
                pickle_saved = pickle.load(file)
            
            for idx, data in enumerate(train_loader):
                input, target = data
                X_test = from_2d_array_to_nested(input.squeeze().cpu().detach().numpy())
                logit_t_np = pickle_saved.predict_proba(X_test)
                logit_t = torch.as_tensor(logit_t_np, dtype = torch.float, device = config.device)
                teacher_list.append(logit_t)

            for idx, data in enumerate(val_loader):
                input, target = data
                X_test = from_2d_array_to_nested(input.squeeze().cpu().detach().numpy())
                logit_t_np = pickle_saved.predict_proba(X_test)
                logit_t = torch.as_tensor(logit_t_np, dtype = torch.float, device = config.device)
                teacher_val_list.append(logit_t)
     
    module_list.to(config.device)
    criterion_list.to(config.device)

    start_training = time.time()
       
    for epoch in range(1, config.epochs + 1):
        reward = train_distilled(epoch, train_loader, module_list, criterion_list, optimizer, 
                                 config, teacher_probs, teacher_list)
        if config.evaluation == 'reinforced':
            #teacher_probs -= reward * config.lr
            teacher_probs -= torch.nan_to_num(reward/torch.max(reward)) * config.lr
            teacher_probs = torch.softmax(teacher_probs, dim=-1)
        if (epoch) % 100 == 0:
            training_time = time.time() - start_training
            accuracy = evaluate(test_loader, model_s, config, epoch, training_time)   
        elif config.pid == 0:
            training_time = 0
            accuracy = evaluate(test_loader, model_s, config, epoch, training_time)

def AED(model, config, teachers):
    config.teachers = len(teachers)

    model_s = model
    model_s.eval()
    model_s = model_s.to(config.device)
    params = list((model_s.parameters()))

    module_list = nn.ModuleList([])
    module_list.append(model_s)

    criterion_list = nn.ModuleList([])

    if config.num_classes > 1:
        criterion_list.append(nn.CrossEntropyLoss())
        criterion_list.append(DistillKL(config.kd_temperature))
    else:
        criterion_list.append(nn.BCEWithLogitsLoss())
        criterion_list.append(DistillKLOne(config.kd_temperature))
    

    if config.teacher_type == 'Inception':
        train_loader, val_loader, test_loader = get_loaders(config)
    else:
        config.batch_size = 10000
        train_loader, val_loader, test_loader = get_loaders(config)
        
    # Teachers
    if config.teacher_type == 'Inception':
        for teacher in teachers:
            savepath = Path('./teachers/Inception_' + config.dataset + '_' + str(teacher) + '_teacher.pkl')
            teacher_config = copy.deepcopy(config)
            teacher_config.bit1 = teacher_config.bit2 = teacher_config.bit3 = 32
            teacher_config.layer1 = teacher_config.layer2 = teacher_config.layer3 = 3
            model_t = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                           bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                           num_pred_classes=config.num_classes,config=teacher_config)

            model_t.load_state_dict(torch.load(savepath, map_location=config.device))
            model_t.eval()
            model_t = model_t.to(config.device)
            module_list.append(model_t)
    else:
        teacher_list = []
        teacher_val_list = []
        for teacher in teachers:
            savepath = Path('./teachers/'+ config.teacher_type + '_' + config.dataset + '_' + str(teacher) + '_teacher.pkl')
            with open(savepath,'rb') as file:
                pickle_saved = pickle.load(file)
            
            for idx, data in enumerate(train_loader):
                input, target = data
                X_test = from_2d_array_to_nested(input.squeeze().cpu().detach().numpy())
                logit_t_np = pickle_saved.predict_proba(X_test)
                logit_t = torch.as_tensor(logit_t_np, dtype = torch.float, device = config.device)
                teacher_list.append(logit_t)

            for idx, data in enumerate(val_loader):
                input, target = data
                X_test = from_2d_array_to_nested(input.squeeze().cpu().detach().numpy())
                logit_t_np = pickle_saved.predict_proba(X_test)
                logit_t = torch.as_tensor(logit_t_np, dtype = torch.float, device = config.device)
                teacher_val_list.append(logit_t)

    teacher_weights = torch.full((1,config.teachers), 1/config.teachers, dtype=torch.float32, 
                                 device = config.device).squeeze()
    
    
    weights_model = TeacherWeights(config, teacher_weights)
    module_list.append(weights_model)
    params.extend(list(weights_model.parameters()))
    optimizer = torch.optim.Adam(model_s.parameters(), lr=config.lr)
    optimizer_w = torch.optim.SGD(weights_model.parameters(), lr=config.lr_w) #Adam ignores the bi-level
        
    module_list.to(config.device)
    criterion_list.to(config.device)

    start_training = time.time()
 
    for epoch in range(1, config.epochs + 1):
        train_distilled(epoch, train_loader, module_list, criterion_list, optimizer, config, [], [])
        
        if (epoch % config.val_epochs == 0) and config.evaluation == 'lightts':
            teacher_weights = validation(epoch, val_loader, module_list, criterion_list, optimizer_w, config)

        if (epoch) % 100 == 0:
            training_time = time.time() - start_training
            accuracy = evaluate(test_loader, model_s, config, epoch, training_time)
        elif config.pid == 0:
            training_time = 0
            teacher_weights = validation(epoch, val_loader, module_list, criterion_list, optimizer_w, config)
            accuracy = evaluate(test_loader, model_s, config, epoch, training_time)
            
    return accuracy, dict(zip(teachers, teacher_weights))

def remove_elements(x):
    return [[el for el in x if el!=x[i]] for i in range(len(x))]

def recursive_accuracy(model,config,max_accuracy,current_teachers):
    subgroups = remove_elements(current_teachers)
    for subgroup in subgroups:
        pivot_accuracy, _ = AED(model, config, subgroup)
        if pivot_accuracy > max_accuracy:
            max_accuracy = pivot_accuracy
            if len(subgroup) > 2:
                recursive_accuracy(model,config,max_accuracy,subgroup)
    return max_accuracy # The value is not updated, so the recursivity continues

def recursive_weight(model,config,teacher_dic):

    if len(list(teacher_dic.keys())) == 10 and config.gumbel == -1.0:
        shuffled = teacher_dic
        random.shuffle(shuffled)
        ordered_weights = list(zip(shuffled.keys(), shuffled.values()))
    elif len(list(teacher_dic.keys())) < 10 and config.gumbel == -1.0:
        ordered_weights = list(zip(teacher_dic.keys(), teacher_dic.values()))
    else:
        ordered_weights = sorted(teacher_dic.items(), key=lambda x: x[1], reverse=False)

    
    if len(list(teacher_dic.keys())) > 8 and config.explore_branches > 1:
        for i in range(0,config.explore_branches):
            copy_weights = copy.deepcopy(teacher_dic)
            del copy_weights[ordered_weights[i][0]]
            new_teachers = list(copy_weights.keys())
            _, new_weights = AED(model, config, new_teachers)
            accuracy = recursive_weight(model,config,new_weights)
    else:
        #remove_key = min(teacher_dic.keys(), key=lambda k: teacher_dic[k])
        del teacher_dic[ordered_weights[0][0]]
        new_teachers = list(teacher_dic.keys())
        accuracy, new_weights = AED(model, config, new_teachers)
        if len(new_teachers) > 2:
            accuracy = recursive_weight(model,config,new_weights)
    return accuracy

def StudentDistillation(model, config):
    teachers = [i for i in range(0,config.teachers)]
    max_accuracy, teacher_weights = AED(model, config, teachers)
    
    if config.evaluation == 'aed-loo':
        recursive_accuracy(model, config, max_accuracy, teachers)
    elif config.evaluation == 'lightts':
        recursive_weight(model, config, teacher_weights)

def RunStudent(config):
    config.layer1 = config.layer2 = config.layer3 = 3
    model_s = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                   bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                   num_pred_classes=config.num_classes,config=config)
    model_s = model_s.to(config.device)

    if config.evaluation == 'lightts':
        StudentDistillation(model_s, config)
    elif config.evaluation == 'aed-loo':
        StudentDistillation(model_s, config)
    elif config.evaluation == 'fp-student':
        fp_config = config
        fp_config.bits = 32
        fp_config.bit1 = fp_config.bit2 = fp_config.bit3 = fp_config.bits
        fp_config.layer1 = fp_config.layer2 = fp_config.layer3 = 3
        model_fp = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                       bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                       num_pred_classes=config.num_classes,config=fp_config)
        model_fp = model_fp.to(config.device)
        config.evaluation = 'lightts'
        StudentDistillation(model_fp, fp_config)
    elif config.evaluation == 'non-distilled-student':
        BasicStudent(model_s, config)
    else:
        BaselineStudent(model_s, config) #'classic','agree','cawpe','reinforced'