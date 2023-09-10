import numpy as np

def Zone(p):
    x = p[0]
    y = p[1]
    # First half
    if (x<=18) and (y<18): return 1
    if (x<=18) and (y>=18) and (y<=62): return 2
    if (x<=18) and (y>62): return 3
    if (x>18) and (x<=39) and (y<18): return 4
    if (x>18) and (x<=39) and (y>=18) and (y<30): return 5
    if (x>18) and (x<=39) and (y>=30) and (y<=50): return 6
    if (x>18) and (x<=39) and (y>50) and (y<=62): return 7
    if (x>18) and (x<=39) and (y>62): return 8
    if (x>39) and (x<=60) and (y<18): return 9
    if (x>39) and (x<=60) and (y>=18) and (y<30): return 10
    if (x>39) and (x<=60) and (y>=30) and (y<=50): return 11
    if (x>39) and (x<=60) and (y>50) and (y<=62): return 12
    if (x>39) and (x<=60) and (y>62): return 13
    # Second half
    if (x>60) and (x<81) and (y<18): return 14
    if (x>60) and (x<81) and (y>=18) and (y<30): return 15
    if (x>60) and (x<81) and (y>=30) and (y<=50): return 16
    if (x>60) and (x<81) and (y>50) and (y<=62): return 17
    if (x>60) and (x<81) and (y>62): return 18
    if (x>=81) and (x<102) and (y<18): return 19
    if (x>=81) and (x<102) and (y>=18) and (y<30): return 20
    if (x>=81) and (x<102) and (y>=30) and (y<=50): return 21
    if (x>=81) and (x<102) and (y>50) and (y<=62): return 22
    if (x>=81) and (x<102) and (y>62): return 23
    if (x>=102) and (x<111) and (y<18): return 24
    if (x>=111) and (y<18): return 25
    if (x>=102) and (y>=18) and (y<30): return 26
    if (x>=102) and (y>=30) and (y<=50): return 27
    if (x>=102) and (y>50) and (y<=62): return 28
    if (x>=102) and (x<111) and (y>62): return 29
    if (x>=111) and (y>62): return 30

def Zone_batch(p):
    c = np.zeros(p.shape[0], dtype=int)
    x = p[:, 0]
    y = p[:, 1]
    # First half
    c[(x<=18) * (y<18)] = 1
    c[(x<=18) * (y>=18) * (y<=62)] = 2
    c[(x<=18) * (y>62)] = 3
    c[(x>18) * (x<=39) * (y<18)] = 4
    c[(x>18) * (x<=39) * (y>=18) * (y<30)] = 5
    c[(x>18) * (x<=39) * (y>=30) * (y<=50)] = 6
    c[(x>18) * (x<=39) * (y>50) * (y<=62)] = 7
    c[(x>18) * (x<=39) * (y>62)] = 8
    c[(x>39) * (x<=60) * (y<18)] = 9
    c[(x>39) * (x<=60) * (y>=18) * (y<30)] = 10
    c[(x>39) * (x<=60) * (y>=30) * (y<=50)] = 11
    c[(x>39) * (x<=60) * (y>50) * (y<=62)] = 12
    c[(x>39) * (x<=60) * (y>62)] = 13
    # Second half
    c[(x>60) * (x<81) * (y<18)] = 14
    c[(x>60) * (x<81) * (y>=18) * (y<30)] = 15
    c[(x>60) * (x<81) * (y>=30) * (y<=50)] = 16
    c[(x>60) * (x<81) * (y>50) * (y<=62)] = 17
    c[(x>60) * (x<81) * (y>62)] = 18
    c[(x>=81) * (x<102) * (y<18)] = 19
    c[(x>=81) * (x<102) * (y>=18) * (y<30)] = 20
    c[(x>=81) * (x<102) * (y>=30) * (y<=50)] = 21
    c[(x>=81) * (x<102) * (y>50) * (y<=62)] = 22
    c[(x>=81) * (x<102) * (y>62)] = 23
    c[(x>=102) * (x<111) * (y<18)] = 24
    c[(x>=111) * (y<18)] = 25
    c[(x>=102) * (y>=18) * (y<30)] = 26
    c[(x>=102) * (y>=30) * (y<=50)] = 27
    c[(x>=102) * (y>50) * (y<=62)] = 28
    c[(x>=102) * (x<111) * (y>62)] = 29
    c[(x>=111) * (y>62)] = 30
    return c