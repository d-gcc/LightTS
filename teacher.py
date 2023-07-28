import numpy as np, torch, time, os
from utils.data import get_loaders, get_raw_data
from utils.inception import InceptionModel
from utils.trainer import train_single, train_distilled, validation, evaluate, evaluate_ensemble

def NN_Teacher(model, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_loader, val_loader, test_loader = get_loaders(config)
    best_accuracy = 0
    start_training = time.time()
    
    for epoch in range(1, config.epochs + 1):
        train_single(epoch, train_loader, model, optimizer, config)
    
        if (epoch) % 100 == 0:
            training_time = time.time() - start_training
            current_accuracy = evaluate(test_loader, model, config, epoch, training_time)
            
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                if not os.path.exists('./teachers/'):
                    os.makedirs('./teachers/')
                model_name = f'Inception_{config.dataset}_{config.init_seed}_teacher.pkl'
                savepath = "./teachers/" + model_name
                torch.save(model.state_dict(), savepath)
                
def SK_Teacher(config):
    from sktime.datatypes._panel._convert import from_2d_array_to_nested
    training, testing = get_raw_data(config)
    
    X_train = from_2d_array_to_nested(training.x.squeeze().cpu().detach().numpy())
    y_train = training.y.squeeze().cpu().detach().numpy()
    
    X_test = from_2d_array_to_nested(testing.x.squeeze().cpu().detach().numpy())
    y_test = testing.y.squeeze().cpu().detach().numpy()

    if config.teacher_type == 'CIF':
        from sktime.classification.interval_based import CanonicalIntervalForest
        classifier = CanonicalIntervalForest(random_state=config.init_seed)
    elif config.teacher_type == 'Forest':
        from sktime.classification.interval_based import TimeSeriesForestClassifier
        classifier = TimeSeriesForestClassifier(random_state=config.init_seed)
    elif config.teacher_type == 'Proximity':
        from sktime.classification.distance_based import ProximityTree
        classifier = ProximityTree(random_state=config.init_seed)
    elif config.teacher_type == 'TDE':
        from sktime.classification.dictionary_based import IndividualTDE
        classifier = IndividualTDE(random_state=config.init_seed)
    elif config.teacher_type == 'Rocket':
        from sktime.classification.kernel_based import RocketClassifier
        classifier = RocketClassifier(random_state=config.init_seed)
    elif config.teacher_type == 'Matrix':
        from sktime.classification.feature_based import MatrixProfileClassifier
        classifier = MatrixProfileClassifier(random_state=config.init_seed)
    
    classifier.fit(X_train, y_train)
    
    model_name = f'{config.teacher_type}_{config.dataset}_{config.init_seed}_teacher.pkl'
    savepath = "./teachers/" + model_name
    
    with open(savepath,'wb') as file:
        pickle.dump(classifier,file)
        
def TeacherEvaluation(config):
    if config.teacher_type == 'Inception':
        _, _, test_loader = get_loaders(config)
    else:
        config.batch_size = 10000
        _, _, test_loader = get_loaders(config)
    evaluate_ensemble(test_loader, config)
    
def RunTeacher(config):    
    if config.evaluation != 'fp-ensemble' and config.teacher_type == 'Inception':
        teacher_config = config
        teacher_config.bits = 32
        teacher_config.bit1 = teacher_config.bit2 = teacher_config.bit3 = teacher_config.bits
        teacher_config.layer1 = teacher_config.layer2 = teacher_config.layer3 = 3
        model_t = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                       bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                       num_pred_classes=config.num_classes,config=teacher_config)
        model_t = model_t.to(config.device)
        
        for teacher in range(0,config.teachers):
            config.init_seed = teacher
            np.random.seed(teacher)
            torch.manual_seed(teacher)
            torch.cuda.manual_seed(teacher)
            torch.backends.cudnn.deterministic = True
            NN_Teacher(model_t, config)
        TeacherEvaluation(config)
    elif config.evaluation != 'fp-ensemble' and config.teacher_type != 'Inception':
        for teacher in range(0,config.teachers):
            config.init_seed = teacher
            np.random.seed(teacher)
            torch.manual_seed(teacher)
            torch.cuda.manual_seed(teacher)
            torch.backends.cudnn.deterministic = True
            SK_Teacher(config)
        TeacherEvaluation(config)
    elif config.evaluation == 'fp-ensemble':
        TeacherEvaluation(config)
