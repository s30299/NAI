from PIL import Image
import numpy as np
import math
from collections import deque
import random
# from PIL.ImagePalette import random
import time
print("Wybierz metryke ")
print("1: Metryka Manhattan ")
print("2: Metryka Euklidesowa ")
print("3: Metryka losowa ")
print("Podaj numer: ")
choice = input()
start_time = time.time()

def search_path_with_astar(start, goal, accessible_fn, h, callback_fn):

    open_set = {tuple(start)}
    closed_set = set()
    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): h(start, goal)}

    while open_set:
        callback_fn(closed_set, open_set)

        # Find the node in open_set with the lowest f_score
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

        if current == tuple(goal):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(tuple(start))
            return path[::-1]  # Return reversed path

        open_set.remove(current)
        closed_set.add(current)

        for neighbor in accessible_fn(current):
            if tuple(neighbor) in closed_set:
                continue

            tentative_g_score = g_score.get(current, float('inf')) + h(neighbor, current)

            if tuple(neighbor) not in open_set:
                open_set.add(tuple(neighbor))
            elif tentative_g_score >= g_score.get(tuple(neighbor), float('inf')):
                continue

            # This path is the best so far
            came_from[tuple(neighbor)] = current
            g_score[tuple(neighbor)] = tentative_g_score
            f_score[tuple(neighbor)] = g_score[tuple(neighbor)] + h(neighbor, goal)

    return []




def h_functionE(a, b):   #Metryka EUKLIDESA
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return math.sqrt(dx ** 2 + dy ** 2)

def h_functionM(a, b):   #Metryka Manhattan
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dx+dy

def h_functionR(a,b):   #Metryka Random
    return random.random()


def getpixel(image, dims, position):
    if any(p < 0 or p >= dims[i] for i, p in enumerate(position)):
        return None
    return image[position[1], position[0]]  # Pillow uses (y, x) indexing


def setpixel(image, dims, position, value):
    if any(p < 0 or p >= dims[i] for i, p in enumerate(position)):
        return
    image[position[1], position[0]] = value


def accessible(bitmap, dims, point):
    neighbors = []
    height, width = dims  # Get the dimensions (height and width)



    # Loop through each dimension (x, y)
    for i in range(len(point)):
        for delta in [-1, 1]:  # Check both directions (left and right, up and down)
            neighbor = list(point)  # Convert tuple to list to modify
            neighbor[i] += delta  # Modify the coordinate
            neighbor = tuple(neighbor)  # Convert back to tuple after modification

            # Ensure the neighbor is within the image bounds
            x, y = neighbor[0], neighbor[1]
            if 0 <= x < width and 0 <= y < height:
                # Ensure it's walkable (pixel value check)
                if bitmap[y, x][0] == 0:  # Assuming the red channel represents walkability (0 = walkable)
                    neighbors.append(neighbor)
    return neighbors


def load_world_map(fname):
    img = Image.open(fname)
    img = img.convert("RGBA")  # Ensure it's in RGBA format
    pixels = np.array(img)
    dims = pixels.shape[:2]  # (height, width)
    return dims, pixels


def save_world_map(fname, image):
    img = Image.fromarray(image)
    img.save(fname)


def find_pixel_position(image, dims, value):
    for y in range(dims[0]):
        print(tuple(image[y, 0]))
        for x in range(dims[1]):
            if tuple(image[y, x]) == value:
                return [x, y]
    raise ValueError("Could not find pixel with the given value!")


if __name__ == "__main__":
    dims, bitmap = load_world_map("img.png")

    start = find_pixel_position(bitmap, dims, (255, 0, 255, 255))  # Cyan pixel 255, 0, 255, 255
    goal = find_pixel_position(bitmap, dims, (255, 255, 0, 255))  # Yellow pixel 255, 255, 0, 255

    setpixel(bitmap, dims, start, (0, 0, 0, 255))
    setpixel(bitmap, dims, goal, (0, 0, 0, 255))



    def on_iteration_nothing(closed_set, open_set):
        pass



    if choice == "1":
        path = search_path_with_astar(start, goal, lambda p: accessible(bitmap, dims, p), h_functionM, on_iteration_nothing)
    if choice == "2":
        path = search_path_with_astar(start, goal, lambda p: accessible(bitmap, dims, p), h_functionE, on_iteration_nothing)
    if choice == "3":
        path = search_path_with_astar(start, goal, lambda p: accessible(bitmap, dims, p), h_functionR, on_iteration_nothing)
    print(path)
    for p in path:
        setpixel(bitmap, dims, p, (255, 0, 0, 255))  # Mark the path with red

    print(len(path))
    save_world_map("result.png", bitmap)
    end_time = time.time()
    print(end_time-start_time)