#!/usr/bin/env python

import rospy
# import math
import rospkg
import numpy as np

import pickle
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from std_srvs.srv import Trigger, TriggerResponse
from controller_manager_msgs.srv import *
from metacontrol_msgs.srv import QAPredictions, QAPredictionsResponse
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class QA_updater(object):
	def __init__(self):
		rospy.Service('/qa_pred_update', QAPredictions, self.qa_update)
		rospy.Service('/load_safety_model', LoadController, self.load_safety_model)

		self._sub_diagnostics = rospy.Subscriber('/diagnostics', DiagnosticArray, self.callbackDiagnostics)
		self._pub_diag = rospy.Publisher(
            '/diagnostics', DiagnosticArray, queue_size=100)

		# Models:
		self._configs = []
		self._models = dict()
		self._models['safety'] = dict()

		self._EMs = ["obstacle_density","narrowness"]
		self._EM_status = dict()
		for EM in self._EMs:
			self._EM_status[EM] = 0

	def qa_update(self,request):
		print('QA update request received!')
		# configs = self._configs
		# configs.remove(req.current_config)
		qa_attr = "safety"

		X = np.zeros(2)
		idx=0
		for EM in self._EMs:
			X[idx] = self._EM_status[EM]
			idx+=1
		X = X.reshape(1, -1)
		pf = PolynomialFeatures(degree=5)
		Xp = pf.fit_transform(X)
		
		values = []
		for config in self._configs:
			# try:
			safety = self._models[qa_attr][config].predict(Xp)[0]

			print("config: {0}, safety:{1}".format(config,safety))

			values.append(
				KeyValue(config, str(safety)))
			# except Exception as exc:
			# 	print(exc)
			# 	print("%s model for %s is not loaded! This quality attribute cannot be updated for this config")

		return QAPredictionsResponse(values)

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
			print('safety model loaded for %s'%config)
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
	            	# print('received %s observation'%(value.key))
	            	self._EM_status[value.key] = value.value
	            

if __name__ == "__main__":
    rospy.init_node('QA_updater')

    updater = QA_updater()

    rospy.spin()