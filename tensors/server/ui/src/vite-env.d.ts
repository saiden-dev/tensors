/// <reference types="vite/client" />

declare module 'vuetify/styles' {
  const styles: string
  export default styles
}

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<object, object, unknown>
  export default component
}
