import unittest

from gym import spaces

from gym_energyplus.envs.energyplus_model_2ZoneDataCenterHVAC_wEconomizer_Temp import (
    EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp
)
from gym_energyplus.envs.energyplus_model_2ZoneDataCenterHVAC_wEconomizer_Temp_Fan import (
    EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan
)


class TestEnergyPlusModel(unittest.TestCase):

    def test_2ZoneDataCenterHVAC_wEconomizer_Temp(self):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp(
            model_file="EnergyPlus/Model-22-1-0/2ZoneDataCenterHVAC_wEconomizer_Temp.idf",
            log_dir=None
        )
        self.assertEqual("2ZoneDataCenterHVAC_wEconomizer_Temp", model.model_basename)

        self.assertIsInstance(model.action_space, spaces.Box)
        self.assertEqual((2,), model.action_space.shape)

        self.assertIsInstance(model.observation_space, spaces.Box)
        self.assertEqual((6,), model.observation_space.shape)

        self.assertTupleEqual(
            (22, 1, 0),
            model.energyplus_version
        )
        self.assertEqual(
            "Electricity Demand Rate",
            model.facility_power_output_var_suffix
        )

    def test_2ZoneDataCenterHVAC_wEconomizer_Temp_Eplus_9_3(self):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp(
            model_file="EnergyPlus/Model-9-3-0/2ZoneDataCenterHVAC_wEconomizer_Temp.idf",
            log_dir=None
        )
        self.assertEqual("2ZoneDataCenterHVAC_wEconomizer_Temp", model.model_basename)
        self.assertTupleEqual(
            (9, 3, 0),
            model.energyplus_version
        )
        self.assertEqual(
            "Electric Demand Power",
            model.facility_power_output_var_suffix
        )

    def test_2ZoneDataCenterHVAC_wEconomizer_Temp_Fan(self):
        model = EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp_Fan(
            model_file="EnergyPlus/Model-22-1-0/2ZoneDataCenterHVAC_wEconomizer_Temp_Fan.idf",
            log_dir=None
        )
        self.assertEqual("2ZoneDataCenterHVAC_wEconomizer_Temp_Fan", model.model_basename)

        self.assertIsInstance(model.action_space, spaces.Box)
        self.assertEqual((4,), model.action_space.shape)

        self.assertIsInstance(model.observation_space, spaces.Box)
        self.assertEqual((6,), model.observation_space.shape)
