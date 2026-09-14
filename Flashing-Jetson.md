# Real-World Setup

We mount a [Jetson Orin AGX 64GB](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) on the [Unitree Go2](https://www.unitree.com/go2) robot.

## NVIDIA Jetson AGX Orin Setup

### Flash the System

Prepare the hardware following the [hardware setup guide](https://developer.nvidia.com/embedded/learn/jetson-agx-orin-devkit-user-guide/two_ways_to_set_up_software.html#hardware-setup).

First, complete the L4T BSP flashing process below.

#### SDK Manager Step 1: Select the Device and JetPack Version

Select **Jetson AGX Orin [64GB developer kit version]** and **JetPack 6.2.1**.

**Do NOT select Host Machine.**

![](images/jetson/flash_1.png)

#### SDK Manager Step 2: Select Jetson Linux

Select **only the first entry, Jetson Linux**.

Do **NOT** select Jetson Runtime Components.

![](images/jetson/flash_2.png)

Click **Continue**.

![](images/jetson/flash_3.png)

Click **Create**.

Then click **Continue** and enter your sudo password.

#### SDK Manager Step 3: Flash the Device

![](images/jetson/flash_4.png)

**Important: Choose RUNTIME and select NVME if an SSD is installed.**

![](images/jetson/flash_5.png)

Click **Flash**.

After a while, SDK Manager returns to this page:

![](images/jetson/flash_6.png)

This step usually takes about 5–8 minutes.

#### SDK Manager Step 4: Finish Flashing

![](images/jetson/flash_7.png)

Flashing is complete.

### Install SDK Components

Before installing SDK components, complete the flashing process above and the initial Ubuntu setup.

Do **NOT** run `sudo apt update` or `upgrade` at this stage, as this may cause version mismatch errors later.

Follow the [SDK Manager guide](https://developer.nvidia.com/embedded/learn/jetson-agx-orin-devkit-user-guide/two_ways_to_set_up_software.html#1-how-to-use-sdk-manager-to-flash-l4t-bsp).

#### SDK Manager Step 1: Detect the Device

The AGX Orin does not need to be in Force Recovery Mode. Wait for SDK Manager to detect it.

![](images/jetson/component_1.png)

#### SDK Manager Step 2: Select SDK Components

Select the desired SDK components as shown below. SDK Manager automatically selects the Runtime components they require.

![](images/jetson/component_2.png)

Click **Continue** and enter your sudo password.

#### SDK Manager Step 3: Install SDK Components

Configure the installation as shown below. Enter the Ubuntu username and password you set on the Jetson AGX Orin:

![](images/jetson/component_3.png)

Click **Install**:

![](images/jetson/component_4.png)

**Troubleshooting: Device image version mismatch**

If SDK Manager reports a device image version mismatch, follow [this guide](https://forums.developer.nvidia.com/t/error-installing-jetpack-6-2-1/337555/5).
This may be caused by running `sudo apt update` and `upgrade` during the initial Ubuntu setup. You can reflash the Jetson AGX Orin to resolve the mismatch.

![](images/jetson/component_5.png)

#### SDK Manager Step 4: Finish Installation

![](images/jetson/component_6.png)