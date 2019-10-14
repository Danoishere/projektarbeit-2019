from deliverables.network import AC_Network

model = AC_Network(True, 'http://127.0.0.1:5000')
model.update_from_global_model()