# simulation.py
import random
import math
import time
import threading
import pygame
import sys
import os

# ── Signal timing defaults ─────────────────────────────
defaultRed     = 150
defaultYellow  = 5
defaultGreen   = 20
defaultMinimum = 10
defaultMaximum = 60

signals        = []
noOfSignals    = 4
simTime        = 300
timeElapsed    = 0

currentGreen  = 0
nextGreen     = (currentGreen + 1) % noOfSignals
currentYellow = 0

# ── Vehicle timing ─────────────────────────────────────
carTime      = 2
bikeTime     = 1
rickshawTime = 2.25
busTime      = 2.5
truckTime    = 2.5
noOfLanes    = 2
detectionTime = 5

# ── Speeds ─────────────────────────────────────────────
speeds = {
    'car': 2.25, 'bus': 1.8, 'truck': 1.8,
    'rickshaw': 2, 'bike': 2.5
}

# ── Start coordinates (based on full 1400x800 canvas) ──
x = {
    'right': [0, 0, 0],
    'down':  [755, 727, 697],
    'left':  [1400, 1400, 1400],
    'up':    [602, 627, 657]
}
y = {
    'right': [348, 370, 398],
    'down':  [0, 0, 0],
    'left':  [498, 466, 436],
    'up':    [800, 800, 800]
}

vehicles = {
    'right': {0: [], 1: [], 2: [], 'crossed': 0},
    'down':  {0: [], 1: [], 2: [], 'crossed': 0},
    'left':  {0: [], 1: [], 2: [], 'crossed': 0},
    'up':    {0: [], 1: [], 2: [], 'crossed': 0}
}

vehicleTypes     = {0: 'car', 1: 'bus', 2: 'truck',
                    3: 'rickshaw', 4: 'bike'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# ── UI coordinates (full canvas 1400x800) ─────────────
signalCoods       = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods  = [(530,210),(810,210),(810,550),(530,550)]
vehicleCountCoods = [(480,210),(880,210),(880,550),(480,550)]

stopLines   = {'right':590,'down':330,'left':800,'up':535}
defaultStop = {'right':580,'down':320,'left':810,'up':545}
stops = {
    'right': [580,580,580],
    'down':  [320,320,320],
    'left':  [810,810,810],
    'up':    [545,545,545]
}

mid = {
    'right': {'x':705,'y':445},
    'down':  {'x':695,'y':450},
    'left':  {'x':695,'y':425},
    'up':    {'x':695,'y':400}
}

rotationAngle = 3
gap  = 15
gap2 = 15

# ── Shared detection data ──────────────────────────────
detection_data = {
    'counts':           {},
    'green_lane':       'right',
    'green_duration':   20,
    'congestion':       'CLEAR',
    'priority_vehicle': False,   # ✅ fixed from 'ambulance'
    'total':            0,
    'fps':              0,
    'latency':          0
}
detection_lock = threading.Lock()

def update_detection(data):
    with detection_lock:
        detection_data.update(data)

pygame.init()
simulation = pygame.sprite.Group()


class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red           = red
        self.yellow        = yellow
        self.green         = green
        self.minimum       = minimum
        self.maximum       = maximum
        self.signalText    = "30"
        self.totalGreenTime = 0


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass,
                 direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane             = lane
        self.vehicleClass     = vehicleClass
        self.speed            = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction        = direction
        self.x                = x[direction][lane]
        self.y                = y[direction][lane]
        self.crossed          = 0
        self.willTurn         = will_turn
        self.turned           = 0
        self.rotateAngle      = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1

        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.originalImage = pygame.image.load(path)
        self.currentImage  = pygame.image.load(path)

        if direction == 'right':
            if (len(vehicles[direction][lane]) > 1 and
                    vehicles[direction][lane][self.index-1].crossed == 0):
                self.stop = (
                    vehicles[direction][lane][self.index-1].stop
                    - vehicles[direction][lane][self.index-1]
                    .currentImage.get_rect().width - gap)
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane]     -= temp
            stops[direction][lane] -= temp

        elif direction == 'left':
            if (len(vehicles[direction][lane]) > 1 and
                    vehicles[direction][lane][self.index-1].crossed == 0):
                self.stop = (
                    vehicles[direction][lane][self.index-1].stop
                    + vehicles[direction][lane][self.index-1]
                    .currentImage.get_rect().width + gap)
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane]     += temp
            stops[direction][lane] += temp

        elif direction == 'down':
            if (len(vehicles[direction][lane]) > 1 and
                    vehicles[direction][lane][self.index-1].crossed == 0):
                self.stop = (
                    vehicles[direction][lane][self.index-1].stop
                    - vehicles[direction][lane][self.index-1]
                    .currentImage.get_rect().height - gap)
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane]     -= temp
            stops[direction][lane] -= temp

        elif direction == 'up':
            if (len(vehicles[direction][lane]) > 1 and
                    vehicles[direction][lane][self.index-1].crossed == 0):
                self.stop = (
                    vehicles[direction][lane][self.index-1].stop
                    + vehicles[direction][lane][self.index-1]
                    .currentImage.get_rect().height + gap)
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane]     += temp
            stops[direction][lane] += temp

        simulation.add(self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))

    def move(self):
        if self.direction == 'right':
            if (self.crossed == 0 and
                    self.x + self.currentImage.get_rect().width
                    > stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if not self.willTurn:
                if ((self.x + self.currentImage.get_rect().width
                        <= self.stop or self.crossed == 1
                        or (currentGreen == 0 and currentYellow == 0))
                        and (self.index == 0
                        or self.x + self.currentImage.get_rect().width
                        < vehicles[self.direction][self.lane]
                        [self.index-1].x - gap2
                        or vehicles[self.direction][self.lane]
                        [self.index-1].turned == 1)):
                    self.x += self.speed

        elif self.direction == 'down':
            if (self.crossed == 0 and
                    self.y + self.currentImage.get_rect().height
                    > stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if not self.willTurn:
                if ((self.y + self.currentImage.get_rect().height
                        <= self.stop or self.crossed == 1
                        or (currentGreen == 1 and currentYellow == 0))
                        and (self.index == 0
                        or self.y + self.currentImage.get_rect().height
                        < vehicles[self.direction][self.lane]
                        [self.index-1].y - gap2
                        or vehicles[self.direction][self.lane]
                        [self.index-1].turned == 1)):
                    self.y += self.speed

        elif self.direction == 'left':
            if (self.crossed == 0 and
                    self.x < stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if not self.willTurn:
                if ((self.x >= self.stop or self.crossed == 1
                        or (currentGreen == 2 and currentYellow == 0))
                        and (self.index == 0
                        or self.x
                        > vehicles[self.direction][self.lane]
                        [self.index-1].x
                        + vehicles[self.direction][self.lane]
                        [self.index-1].currentImage.get_rect().width
                        + gap2
                        or vehicles[self.direction][self.lane]
                        [self.index-1].turned == 1)):
                    self.x -= self.speed

        elif self.direction == 'up':
            if (self.crossed == 0 and
                    self.y < stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if not self.willTurn:
                if ((self.y >= self.stop or self.crossed == 1
                        or (currentGreen == 3 and currentYellow == 0))
                        and (self.index == 0
                        or self.y
                        > vehicles[self.direction][self.lane]
                        [self.index-1].y
                        + vehicles[self.direction][self.lane]
                        [self.index-1].currentImage.get_rect().height
                        + gap2
                        or vehicles[self.direction][self.lane]
                        [self.index-1].turned == 1)):
                    self.y -= self.speed


def initialize():
    for i in range(noOfSignals):
        if i == 0:
            signals.append(TrafficSignal(
                0, defaultYellow, defaultGreen,
                defaultMinimum, defaultMaximum))
        else:
            signals.append(TrafficSignal(
                defaultRed, defaultYellow, defaultGreen,
                defaultMinimum, defaultMaximum))
    repeat()


def setTime():
    global currentGreen
    with detection_lock:
        d = detection_data.copy()
    counts = d.get('counts', {})
    cars   = counts.get('car',           0)
    buses  = counts.get('bus',           0)
    trucks = counts.get('truck',         0)
    bikes  = counts.get('motorcycle',    0)
    ricks  = counts.get('auto-rickshaw', 0)
    t = math.ceil(
        (cars*carTime + buses*busTime + trucks*truckTime
         + bikes*bikeTime + ricks*rickshawTime)
        / (noOfLanes + 1)
    )
    t = max(defaultMinimum, min(t, defaultMaximum))
    signals[currentGreen].green = t


def repeat():
    global currentGreen, currentYellow, nextGreen
    while signals[currentGreen].green > 0:
        printStatus()
        updateValues()
        if signals[currentGreen].green == detectionTime:
            t = threading.Thread(name="detection", target=setTime)
            t.daemon = True
            t.start()
        time.sleep(1)

    currentYellow = 1
    for i in range(3):
        stops[directionNumbers[currentGreen]][i] = \
            defaultStop[directionNumbers[currentGreen]]
        for v in vehicles[directionNumbers[currentGreen]][i]:
            v.stop = defaultStop[directionNumbers[currentGreen]]

    while signals[currentGreen].yellow > 0:
        printStatus()
        updateValues()
        time.sleep(1)

    currentYellow = 0
    signals[currentGreen].green  = defaultGreen
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red    = defaultRed

    currentGreen = nextGreen
    nextGreen    = (currentGreen + 1) % noOfSignals
    signals[nextGreen].red = (signals[currentGreen].yellow
                              + signals[currentGreen].green)
    repeat()


def printStatus():
    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                print(f"GREEN TS{i+1} → g:{signals[i].green}")
            else:
                print(f"YELLOW TS{i+1} → y:{signals[i].yellow}")
        else:
            print(f"RED TS{i+1} → r:{signals[i].red}")
    print()


def updateValues():
    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green         -= 1
                signals[i].totalGreenTime += 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1


def generateVehicles():
    while True:
        vehicle_type = random.randint(0, 4)
        lane_number  = 0 if vehicle_type == 4 \
                       else random.randint(0, 1) + 1
        will_turn = 0
        if lane_number == 2:
            will_turn = 1 if random.randint(0, 4) <= 2 else 0
        temp = random.randint(0, 999)
        a    = [400, 800, 900, 1000]
        direction_number = 0
        if   temp < a[0]: direction_number = 0
        elif temp < a[1]: direction_number = 1
        elif temp < a[2]: direction_number = 2
        else:             direction_number = 3
        Vehicle(lane_number,
                vehicleTypes[vehicle_type],
                direction_number,
                directionNumbers[direction_number],
                will_turn)
        time.sleep(0.75)


def simulationTime():
    global timeElapsed, simTime
    while True:
        timeElapsed += 1
        time.sleep(1)
        if timeElapsed == simTime:
            total = 0
            print('Lane-wise Vehicle Counts:')
            for i in range(noOfSignals):
                cnt = vehicles[directionNumbers[i]]['crossed']
                print(f'  Lane {i+1}: {cnt}')
                total += cnt
            print(f'Total passed: {total}')
            os._exit(1)


def run_simulation():

    threading.Thread(name="simTime",
        target=simulationTime, daemon=True).start()
    threading.Thread(name="init",
        target=initialize, daemon=True).start()
    threading.Thread(name="genVehicles",
        target=generateVehicles, daemon=True).start()

    # ── Canvas = full 1400x800 (all coords stay the same) ─
    CANVAS_W = 1400
    CANVAS_H = 800

    # ── Window = half screen, positioned LEFT ─────────────
    WIN_W = 700
    WIN_H = 500

    os.environ['SDL_VIDEO_WINDOW_POS'] = '0,30'  # left side

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Smart Traffic Simulation")

    # Full-size virtual surface — everything draws here
    canvas = pygame.Surface((CANVAS_W, CANVAS_H))

    background   = pygame.image.load('images/mod_int.png')
    redSignal    = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal  = pygame.image.load('images/signals/green.png')
    font         = pygame.font.Font(None, 30)
    font_big     = pygame.font.Font(None, 40)

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # ── Draw everything onto full-size canvas ──────────
        canvas.blit(background, (0, 0))

        # Signals
        for i in range(noOfSignals):
            if i == currentGreen:
                if currentYellow == 1:
                    signals[i].signalText = signals[i].yellow
                    canvas.blit(yellowSignal, signalCoods[i])
                else:
                    signals[i].signalText = signals[i].green
                    canvas.blit(greenSignal, signalCoods[i])
            else:
                signals[i].signalText = (
                    signals[i].red if signals[i].red <= 10 else "---")
                canvas.blit(redSignal, signalCoods[i])

        # Timers + vehicle counts
        white = (255, 255, 255)
        black = (0, 0, 0)
        green = (0, 255, 0)
        for i in range(noOfSignals):
            txt  = font.render(str(signals[i].signalText),
                               True, white, black)
            canvas.blit(txt, signalTimerCoods[i])
            cnt  = vehicles[directionNumbers[i]]['crossed']
            vtxt = font.render(str(cnt), True, black, white)
            canvas.blit(vtxt, vehicleCountCoods[i])

        # Time elapsed
        te_txt = font.render(
            f"Time: {timeElapsed}s", True, black, white)
        canvas.blit(te_txt, (1100, 50))

        # Detection panel
        with detection_lock:
            d = detection_data.copy()

        pygame.draw.rect(canvas, (20, 20, 20), (0, 0, 420, 130))
        pygame.draw.rect(canvas, (50, 50, 50), (0, 0, 420, 130), 2)

        cong = d.get('congestion', 'CLEAR')
        cong_color = (0,200,0) if cong=='CLEAR' \
               else (255,165,0) if cong=='MODERATE' \
               else (255,0,0)

        canvas.blit(font_big.render(
            "YOLO Detection", True, (100,200,255)), (10, 8))

        counts_str = "  ".join(
            f"{k[:3].upper()}:{v}"
            for k, v in d.get('counts', {}).items()
            if v > 0
        ) or "No vehicles"
        canvas.blit(font.render(counts_str, True, white), (10, 50))
        canvas.blit(font.render(
            f"Status: {cong}  |  "
            f"FPS:{d.get('fps',0):.0f}  |  "
            f"Lat:{d.get('latency',0):.0f}ms",
            True, cong_color), (10, 80))
        canvas.blit(font.render(
            f"Green: {d.get('green_lane','?').upper()}  |  "
            f"Duration: {d.get('green_duration',0)}s",
            True, green), (10, 108))

        # ✅ Priority vehicle alert (was ambulance)
        if d.get('priority_vehicle', False):
            pygame.draw.rect(canvas, (150,0,0),
                (0, CANVAS_H-60, CANVAS_W, 60))
            canvas.blit(font_big.render(
                "*** PRIORITY VEHICLE — GREEN WAVE ACTIVE ***",
                True, white), (150, CANVAS_H-45))

        # Draw vehicles onto canvas
        for vehicle in simulation:
            canvas.blit(vehicle.currentImage, (vehicle.x, vehicle.y))
            vehicle.move()

        # ── Scale full canvas → small window ───────────────
        scaled = pygame.transform.scale(canvas, (WIN_W, WIN_H))
        screen.blit(scaled, (0, 0))

        pygame.display.update()
        clock.tick(30)


if __name__ == '__main__':
    run_simulation()