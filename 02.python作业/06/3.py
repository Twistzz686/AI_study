class Car():
    def __init__(self,color,type):
        self.color = color
        self.type = type
    def move(self):
        print(f"该车的型号为{self.color},颜色为{self.type}")
bmw = Car("red","X1")
audi = Car("blue","a4")
bmw.move()
audi.move()