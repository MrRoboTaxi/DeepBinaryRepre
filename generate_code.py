from operator import *

class CodeGenerator():


    def __init__(self, k, d):
        self.k = k
        self.d = d
        self.set = self.hs(k, d)
        self.len = len(self.set)

    def build_space(self, k): 
        space = {} 
        for i in range(0,2**k): 
            space[i]=1 
        return space 
     
    def build_ball(self, k,d): 
        ball_1 = [] 
        for i in range(0,k): 
            ball_1.append(2**i) 
        ball = {} 
        for i in ball_1: ball[i]=1 
        for i in range(2,d): 
            d_ball={} 
            for j in ball_1: 
                for k in ball.keys(): 
                    d_ball[xor(j,k)] = 1 
            for l in d_ball.keys(): 
                ball[l] = 1 
        if ball.has_key(0): del ball[0] 
        return ball.keys() 
     
    def hs(self, k,d): 
        space = self.build_space(k) 
        ball = self.build_ball(k,d) 
        for i in space.keys(): 
            if space[i]: 
                for j in ball: 
                    space[xor(i,j)]=0 
        min_set = filter(lambda i:space[i], space.keys()) 
        return min_set 

    def get_num(self):
        return self.len

    def get_code(self):
        return self.set

    def get_hash(self, num = 10):
        hash_code = []
        for s in self.set:
            binary = [int(i) for i in list(bin(s)[2:])]
            append = [0 for i in range(self.k - len(binary))]
            hash_code.append(append + binary)
        start = self.len / 2 - num / 2
        return hash_code[start : start + num]

def generate_code_74(length = 48):
    ham74 = [
            [0,0,0,0,0,0,0],
            [0,0,0,1,0,1,1],
            [0,0,1,0,1,0,1],
            [0,0,1,1,1,1,0],
            [0,1,0,0,1,1,0],
            [0,1,0,1,1,0,1],
            [0,1,1,0,0,1,1],
            [0,1,1,1,0,0,0],
            [1,0,0,0,1,1,1],
            [1,0,0,1,1,0,0],
            [1,0,1,0,0,1,0],
            [1,0,1,1,0,0,1],
            [1,1,0,0,0,0,1],
            [1,1,0,1,0,1,0],
            [1,1,1,0,1,0,0],
            [1,1,1,1,1,1,1]]
    m = length / 7
    n = length % 7
    code = [a * m + a[-n:] for a in ham74]
    return code[0:10]

# coder = CodeGenerator(12,6)
# print coder.get_code()
# print coder.get_hash()
# print coder.get_num()