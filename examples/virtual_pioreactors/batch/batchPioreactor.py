
class BatchVirtualPioreacor:

    def __init__(self, vmax, km, y_xs, s_in, x_in):
        self.vmax = vmax
        self.km = km
        self.y_xs = y_xs

        self.s = s_in
        self.x = x_in

        self.time = 0.0

    def step(self, dt):

        s = self.s
        x = self.x

        mu = self.vmax * s / (s + self.km)

        dx_dt = mu * x
        ds_dt = - (mu / self.y_xs) * x

        self.x += dx_dt*dt
        self.s += ds_dt*dt

        self.time += dt
        
        #print(self.time, self.x, self.s, mu)

