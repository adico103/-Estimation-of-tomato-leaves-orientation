import process_utils
import feather
import params_update
import get_contour
import numpy as np

class DataPreProcess:
    def __init__(self,
                    

                    ):
        self.combos_1_path = 'combos_1.feather'
        self.combos_2_path = 'combos_2.feather'


    def create_raw_data(self):
        folders = np.arange(40)+1
        process_utils.creat_complete_file(folders=folders)


    def extract_contour_joints(self):
        get_contour.save_all_joints(params_update.raw_data_directory,N = [1000],joints_folder=params_update.joints_folder)


    def compute_norms(self):
        process_utils.find_normals()

    def combine_all(self):
        process_utils.combine_feathers()

    def make_total_filltered_file(self):
        self.total_data = feather.read_dataframe(params_update.Total_data)
        process_utils.filtering(self.total_data,params_update.Total_filltered_data)
        
    def make_1_image_data(self):
        self.total_filtered_data = feather.read_dataframe(params_update.Total_filltered_data)

        ## making 1 image data

        process_utils.make_reg_data(self.total_filtered_data)

        reg_data = feather.read_dataframe(self.combos_1_path)
        process_utils.prepare_by_joints_num_reg(20,reg_data)
    
    def make_2_imgs_data(self):
        # making 2 images data
        self.total_filtered_data = feather.read_dataframe(params_update.Total_filltered_data)
        process_utils.make_total_filltered_pairs(self.total_filtered_data)
        self.pairs_from_total_filltered = feather.read_dataframe(params_update.Total_filltered_pairs)
        process_utils.combine_n_images(self.pairs_from_total_filltered,2,all_data=self.total_filtered_data)
        combos_2 = feather.read_dataframe(self.combos_2_path)
        process_utils.prepare_by_joints_num(20,combos_2,2)

    def make_3_imgs_data(self):
        # making 3 images data
        self.pairs_from_total_filltered = feather.read_dataframe(params_update.Total_filltered_pairs)
        process_utils.combine_n_images(self.pairs_from_total_filltered,3,all_data=self.total_filtered_data)
        for i in range (10):
            combos_3 = feather.read_dataframe('combos_3_'+str(i+1)+'.feather')
            process_utils.prepare_by_joints_num(20,combos_3,3,sub_num=i+1)


datapreprocess = DataPreProcess()

datapreprocess.create_raw_data()
datapreprocess.extract_contour_joints()
datapreprocess.compute_norms()
datapreprocess.combine_all()
datapreprocess.make_total_filltered_file()
datapreprocess.make_1_image_data()
datapreprocess.make_2_imgs_data()
datapreprocess.make_3_imgs_data()