<template>
  <div>
    <h1> {{ title }}</h1>
    <p> {{ message }}</p>
    <input v-model = "message" placeholder = "Type something" />
    <button @click="updateTitle"> Update Title </button>
  </div>
</template>

<script>

import { ref } from 'vue';

export default {
  name: 'Test',
  setup() {
    const title = ref('Hey there');
    const message = ref('');

    const updateTitle = () => {
      title.value = 'Title Updated!';
    };

    return {
      title,
      message,
      updateTitle
    };
  },
};

</script>


<style scoped>
h1 {
  color: #42b983;
}
</style>