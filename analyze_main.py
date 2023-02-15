from analazys import ExpMaster,compare_best_models,save_final_table,plot_best_elevation_angles,analyze_images_contribution


path='Analayse_data/'
table_path = path+'final_table.csv'


######## get all information ######
experiments = [3,5,7,8,9]
for exp in experiments:
    all_exp = ExpMaster(num_exp = exp)
    all_exp.save_exps_info()

######## saving csv final result table ######

experiments = [3,5,7,8,9]
save_final_table(experiments,table_path,old_exp=False)


######## plot pred vs true for best models ########

errors = [13.5]
exps = [5,3,7]
for error in errors:    
    compare_best_models(table_path,error,exps,show=True)

######## Cumulative propability graphs ########

exps = [3,5]
plot_best_elevation_angles(exps,table_path)


######### analyze models contribution between 1,2,3 images ########

case_studie = [5] #3 image model
exps = [3,7] # 2 image model, 1 image model
recreate_table = True # change to True when there is a new data /exp to analyze
analyze_images_contribution(case_studie,exps,table_path,recreate_table = recreate_table)


