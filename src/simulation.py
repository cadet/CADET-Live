# -*- coding: utf-8 -*-

from CADETProcess.simulator import Cadet
cadet = Cadet()
cadet.run_h5("./data.h5")
cadet.check_cadet()


### main loop


start_mqtt()

while true :
    get_latest_data()
    update_param()
    run_sim()