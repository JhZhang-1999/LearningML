class logGate(object):
    def __init__(self,x,dlogx):
        self.x=x
        self.dlogx=dlogx
    def forward(self):
        logx=math.log(self.x)
        return logx
    def backward(self):
        dx=self.dlogx*(1/self.x)
        return dx

class expGate(object):
    def __init__(self,x,dexpx):
        self.x=x
        self.dexpx=dexpx
    def forward(self):
        expx=math.exp(self.x)
        return expx
    def backward(self):
        dx=self.dexpx
        return dx

class divGate(object):
    def __init__(self,nume,deno):
        self.nume=nume
        self.deno=deno
    def forward(self):
        x_on_y=self.nume/self.deno
        return x_on_y
    def backward(self):
        dx=
