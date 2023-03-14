import matplotlib.pyplot as plt

full_sotf_x = [130, 150, 170]
full_sotf_y = [0.004013653727995381, 0.0026163584046899053, 0.00902862832784632]
control_x = [130, 150, 170]
control_y = [0.014975664809707424, 0.0077571710939721575, 0.0055500193903571925]
lerp_x = [150, 155, 160, 165, 170]
lerp_y = [0.0037181026345809868, 0.004149576186620497, 0.004167941513116632, 0.004817449827049714, 0.003376792499346329]

plt.figure(1)
plt.plot(control_x, control_y, color='k', linestyle='dashed', marker='o',
     markerfacecolor='k')
plt.plot(full_sotf_x, full_sotf_y, color='k', linestyle='dotted', marker='v',
     markerfacecolor='k')
plt.plot(lerp_x, lerp_y, color='xkcd:bright orange', linestyle='dashdot', marker='s',
     markerfacecolor='xkcd:bright orange')
plt.legend(['Control', 'SOTF enabled', 'SOTF to control (linearly interpolated)'])
plt.xlabel('Target fluorescence intensity (arb.)')
plt.ylabel('Standard deviation in refractive index')
plt.ylim([0, 0.02])
plt.gca().tick_params(direction='in')
plt.savefig('lerp_data.png', dpi=600)
plt.show()
