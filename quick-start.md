# Quick Start

## Install Seg2Link

### Install conda

Seg2Link is a python-based software. To use python environment, users could download and install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first. &#x20;

### Create a new conda environment

Open the terminal in macOS or Linux, or open Anaconda/Miniconda prompt in Windows. Then type following command (after the prompt $) and press **Enter** to create a new python environment.&#x20;

{% hint style="info" %}
You can replace the environment name **my-env** with another name that you prefer.
{% endhint %}

```
$ conda create --name my-env python==3.8 pip
```

### Install Seg2Link

Type following commands (after the prompt $) and press **Enter** to activate the created environment and install Seg2Link.

```
$ conda activate my-env
(my-env)$ pip install seg2link
```

{% hint style="info" %}
The pip tool will automatically download and install the package **seg2link** and other prerequisite packages. Depending on your internet speed, this may take a short or long time.
{% endhint %}

## Prepare images for segmentation

### Required images:

1. Predictions of cell/non cell regions\
   \- Used to generate individual cells in each slice.\
   \- Typically this is created by predicting from the raw image with a pre-trained deep neural network A.\
   \- Format: 2D tiff image sequence, stored in a single folder (Typically 8 bit)
2. Raw images\
   \- Used as a reference for uses to correct segmentation mistakes\
   \- Format: 2D tiff image sequence, stored in a single folder (Typically 8 bit)
3. (Optional) Mask images\
   \- Used to ignore the cells that are not of interest\
   \- Typically this is created by predicting from the raw image with another pre-trained deep neural network B.\
   \- Format: 2D tiff image sequence, stored in a single folder (Typically 8 bit)

{% swagger baseUrl="https://api.myapi.com/v1" method="post" path="/pet" summary="Create pet." %}
{% swagger-description %}
Creates a new pet.
{% endswagger-description %}

{% swagger-parameter in="body" name="name" required="true" type="string" %}
The name of the pet
{% endswagger-parameter %}

{% swagger-parameter in="body" name="owner_id" required="false" type="string" %}
The 

`id`

 of the user who owns the pet
{% endswagger-parameter %}

{% swagger-parameter in="body" name="species" required="false" type="string" %}
The species of the pet
{% endswagger-parameter %}

{% swagger-parameter in="body" name="breed" required="false" type="string" %}
The breed of the pet
{% endswagger-parameter %}

{% swagger-response status="200" description="Pet successfully created" %}
```javascript
{
    "name"="Wilson",
    "owner": {
        "id": "sha7891bikojbkreuy",
        "name": "Samuel Passet",
    "species": "Dog",}
    "breed": "Golden Retriever",
}
```
{% endswagger-response %}

{% swagger-response status="401" description="Permission denied" %}

{% endswagger-response %}
{% endswagger %}

{% hint style="info" %}
**Good to know:** You can use the API Method block to fully document an API method. You can also sync your API blocks with an OpenAPI file or URL to auto-populate them.
{% endhint %}

Take a look at how you might call this method using our official libraries, or via `curl`:

{% tabs %}
{% tab title="curl" %}
```
curl https://api.myapi.com/v1/pet  
    -u YOUR_API_KEY:  
    -d name='Wilson'  
    -d species='dog'  
    -d owner_id='sha7891bikojbkreuy'  
```
{% endtab %}

{% tab title="Node" %}
```javascript
// require the myapi module and set it up with your API key
const myapi = require('myapi')(YOUR_API_KEY);

const newPet = away myapi.pet.create({
    name: 'Wilson',
    owner_id: 'sha7891bikojbkreuy',
    species: 'Dog',
    breed: 'Golden Retriever',
})
```
{% endtab %}

{% tab title="Python" %}
```python
// Set your API key before making the request
myapi.api_key = YOUR_API_KEY

myapi.Pet.create(
    name='Wilson',
    owner_id='sha7891bikojbkreuy',
    species='Dog',
    breed='Golden Retriever',
)
```
{% endtab %}
{% endtabs %}