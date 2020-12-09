Generate_Params = \
{   "sample_tokens_path":"/home/xyl/Research project/A-data-processing-mechanism-for-radar-and-camera-data-fusion/data/nuscence/train.txt",
    "radar_sensors":['RADAR_FRONT'],#采用前向雷达
    "camera_sensors":['CAM_FRONT'],#采用前向摄像头
    "radar_cover_shape":"circle",#有三种模式：“normal_point, line, circle”
    "category_mapping":{
            "vehicle.car": "vehicle.car",
            #"vehicle.motorcycle": "vehicle.motorcycle",
            #"vehicle.bicycle": "vehicle.bicycle",
            "vehicle.bus": "vehicle.bus",
            "vehicle.truck": "vehicle.truck",
            "vehicle.emergency": "vehicle.truck",
            "vehicle.trailer": "vehicle.trailer",},
            #"human": "human", },#所要训练的类别，可以把人给去除
    "noisy_image_method":None,#给图片添加噪声
    "noise_factor":0,#噪声的影响因子
    "camera_dropout":0,#视觉信息丢弃的概率
    "radar_dropout": 0,#雷达信息丢弃的概率
    "image_radar_fusion": True,#是否进行视觉与雷达的融合，这个其实有点冗余，设置True即可
    "noise_filter":None,#是否过滤掉雷达返回的噪声
    "n_sweeps":5,#每帧雷达数据中需要返回多少次扫描的点
    "perfect_noise_filter":False,#是否通过真实3D框标注来对雷达噪声进行过滤
    "visibilities":['2', '3', '4'],#通过目标的可见程度来筛选目标，'3','4'代表容易看见的等级，不容易看见的物体在‘1’，‘2’等级
    "noise_category_selection":['vehicle.car',
                                         'vehicle.motorcycle',
                                         'vehicle.bicycle',
                                         'vehicle.bus.rigid',
                                         'vehicle.truck',
                                         'vehicle.trailer',
                                         'human.pedestrian.adult'
                                         ],#这里还要根据程序来变动,保留gt种类的radar，其余过滤
    "normalize_radar":True,#对扩展的雷达数据进行标准化，然后扩展到图像的像素值的变化范围
    "channels":[0, 1, 2, 5, 6, 7, 18, 19], #[0, 1, 2, 5, 6, 7, 18, 19],#0,1,2为图像的RGB通道，而后面的则是所挑选的radar属性，作为模型输入的额外通道
    "isdebug":True,#是否进入调试模式
    "image_plus_visual":True,#是否对融合radar的图像进行可视化
}