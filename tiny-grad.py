class Value:
    def __init__(self,data ,_children =() ,_op=''):
        self.data= data
        self.grad =0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda:None

    def __repr__(self):
        return f"Value (data = {self.data}, grad = {self.grad})"
    
    def __add__(self,other):
        out = Value(self.data+other.data,(self,other),'+')

        def _backward():
            self.grad +=1.0 *out.grad
            other.grad+=1.0*out.grad
        out._backward = _backward
        return out 
    
    def __mul__(self,other):
        out= Value(self.data * other.data, (self,other),'*')
        def _backward():
            self.grad+=other.data*out.data
            other.grad+=self.data*out.data
        out._backward = _backward


        return out 
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if  v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)
                topo.append(v)



        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()



a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = Value(2.0)

e = a * b
d = e + c
L = d * f

'''
a=2 ----\
          * ---> e=-6 ----\
b=-3 ---/                  +
                            ---> d=4 ----\
c=10 ---------------------/              *
                                          ---> L=8
f=2 -------------------------------------/
'''

print("Loss:", L)
# Loss = 8.0

L.backward()

print("\n Gradients :")
print("a.grad =", a.grad)
print("b.grad =", b.grad)
print("c.grad =", c.grad)
print("f.grad =", f.grad)

