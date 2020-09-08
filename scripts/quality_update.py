#!/usr/bin/env python

import rospy
# import math
import rospkg
import numpy as np

import pickle
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from std_srvs.srv import Trigger, TriggerResponse
from controller_manager_msgs.srv import *

class QA_updater(object):
	def __init__(self):
		rospy.Service('/safety_update', Trigger, self.safety_update)
		rospy.Service('/load_safety_model', LoadController, self.load_safety_model)

		self._sub_diagnostics = rospy.Subscriber('/diagnostics', DiagnosticArray, self.callbackDiagnostics)

		# Models:
		self._configs = []
		self._models = dict()
		self._models['safety'] = dict()

		self._EMs = ["narrowness", "d_narrowness", 'obstacle_density', "d_obstacle_density"]
		self._EM_status = dict()
		for EM in self._EMs:
			self._EM_status[EM] = 0

	def safety_update(self,req):
		# configs = self._configs
		# configs.remove(req.current_config)
		qa_attr = "safety"

		X = np.zeros(4)
		idx=0
		for EM in self._EMs:
			X[idx] = self._EM_status[EM]
			idx+=1
		X = X.reshape(1, -1)
		y = dict()

		for config in self._configs:
			if config in self._models[qa_attr]:
				y[config] = self._models[qa_attr][config].predict(X)
			else:
				print("%s model for %s is not loaded! This quality attribute cannot be updated for this config")
		return TriggerResponse(
			success=True,
			message=str(y)
			)

	def load_safety_model(self,req):
		loaded = True
		config = req.name
		try:
			rospack = rospkg.RosPack()
			config_path = rospack.get_path(config)
			pkl_model_path = config_path + "/" + "/quality_models/safety.pkl"
			with open(pkl_model_path, 'rb') as file:
				self._models['safety'][config] = pickle.load(file)
				self._configs.append(config)
		except Exception as exc:
			print(exc)
			print("Could not load {}".format(config))
			loaded = False
		return LoadControllerResponse(loaded)

	def callbackDiagnostics(self,msg):
	    for diagnostic_status in msg.status:
	        if diagnostic_status.message == "EM status":
	            # rospy.loginfo('received EM observation')
	            for value in diagnostic_status.values:
	            	rospy.loginfo('received %s observation'%(value.key))
	            	self.EM_status[value.key] = value.value
	            

if __name__ == "__main__":
    rospy.init_node('QA_updater')

    updater = QA_updater()

    rospy.spin()