from TcpWithVM import Tcp

mytcp = Tcp()
mytcp.connect()
mytcp.sendmessa("palm1")
mytcp.sendmessa("palm2")
mytcp.sendmessa("fist")
mytcp.disconnect()
