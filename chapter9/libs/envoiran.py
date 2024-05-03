import gym
import gym.spaces
from gym.utils import seeding
from gym.envs.registration import EnvSpec
import enum
import collections
import numpy as np


DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1
Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2

class State:
    def __init__(self, bars_count, commission_perc,
                 reset_on_close, reward_on_close=True,
                 volumes=True):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    def reset(self, prices, offset):
        assert isinstance(prices, Prices)
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self):
        # [h, l, c] * bars + position_flag + rel_profit
        if self.volumes:
            return 4 * self.bars_count + 1 + 1,
        else:
            return 3*self.bars_count + 1 + 1,

    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            ofs = self._offset + bar_idx
            
            res[shift] = self._prices.high[ofs]
            shift += 1
            res[shift] = self._prices.low[ofs]
            shift += 1
            res[shift] = self._prices.close[ofs]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[ofs]
                shift += 1
            print(f"""state off set ofs {ofs}\n and shape res as batch from offset {res.shape} \n 
                    state_offset {self._offset} \n bar_idx {bar_idx} \n shift {shift} \n res shift {res}""")
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = self._cur_close() / self.open_price - 1.0
        print(f"Final res shape {res.shape} shift {shift}")
        return res

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close / self.open_price - 1.0)
            self.have_position = False
            self.open_price = 0.0

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1

        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close / prev_close - 1.0)

        return reward, done


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    #spec = EnvSpec("StocksEnv-v0",entry_point=libs.envoiran.StocksEnv)

    def __init__(self, prices: Prices, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC,
                 reset_on_close=True, state_1d=False,
                 random_ofs_on_reset=True, reward_on_close=False,
                 volumes=False):
        self._prices = prices
        self._state = State(
            bars_count, commission, reset_on_close,
            reward_on_close=reward_on_close, volumes=volumes)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        
        #self.seed()
    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
    
    def reset(self):
        self._instrument = self.np_random.choice(
            list(self._prices._fields))
        if self._instrument is "open":
            prices = self._prices.open
        if self._instrument is "close":
            prices = self._prices.close
        if self._instrument is "high":
            prices = self._prices.high
        if self._instrument is "low":
            prices = self._prices.low
        else:
            prices = self._prices.volume
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(
                prices.shape[0]-bars*10) + bars
        else:
            offset = bars
        print(self._prices.low[offset],offset)
        
        # return P, offset
        self._state.reset(self._prices, offset)
        return self._state.encode()
    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {

            
                "instrument": self._instrument,
                "offset": self._state._offset
                }
        return obs, reward, done, info